use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::StatusCode;
use axum::{routing::get, routing::post, Json, Router};
use base64::Engine;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

mod catalogue;
mod decision;
mod jev;
mod vision;

use catalogue::Catalogue;
use decision::Thresholds;
use jev::JevConfig;

// Server entry point: routes, app state, and the main recognize/confirm
// flow. The actual vision/decision/catalogue/jev logic lives in their
// own modules, this file mostly wires them together.

struct AppState {
    catalogue: Catalogue,
    thresholds: Thresholds,
    jev: JevConfig,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    if let Err(e) = vision::init("models/mobilenetv2.onnx", "models/imagenet_classes.txt")
    {
        tracing::error!("failed to load vision model: {e}");
        tracing::error!("the server will still start, but /recognize will return 503");
    } else {
        tracing::info!("vision model loaded");
    }

    let db_path = std::env::var("POS_DB_PATH").unwrap_or_else(|_| "pos_vision.sqlite".to_string());
    let catalogue = match Catalogue::open(&db_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("failed to open catalogue db at {db_path}: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("catalogue db ready at {db_path}");

    let thresholds = Thresholds::from_env();
    tracing::info!(
        "decision thresholds: auto_accept>={:.2}, review>={:.2}",
        thresholds.auto_accept,
        thresholds.review
    );

    let jev = JevConfig::from_env();
    tracing::info!(
        "jev experiment: {}",
        if jev.enabled { "enabled" } else { "disabled" }
    );

    let state = Arc::new(AppState {
        catalogue,
        thresholds,
        jev,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/recognize", post(recognize))
        .route("/products", get(list_products))
        .route("/confirm", post(confirm))
        .route("/stats.md", get(stats_md))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .fallback_service(ServeDir::new("static"))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    tracing::info!("pos_vision listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok", "milestone": "M4" }))
}

#[derive(Deserialize)]
struct RecognizeRequest {
    /// Base64-encoded JPEG/PNG frame from the browser canvas.
    image_base64: String,
}

/// Capture -> recognize -> map to catalogue -> decide AUTO_ACCEPT/
/// REVIEW/MANUAL. Logs the event and timing as it goes.
async fn recognize(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RecognizeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let start = Instant::now();

    let image_bytes = base64::engine::general_purpose::STANDARD
        .decode(&payload.image_base64)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid base64: {e}") })),
            )
        })?;

    // Hash the image rather than storing it, keeps the events table
    // small and avoids hanging onto photos we don't need.
    let image_hash = format!("{:x}", Sha256::digest(&image_bytes));

    let predictions = match vision::recognize(&image_bytes, 3) {
        Ok(p) => p,
        Err(vision::VisionError::ModelNotLoaded) => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": "vision model not loaded" })),
            ))
        }
        Err(e) => {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    };

    let top = predictions.first();
    let matched_product = match top {
        Some(p) => state
            .catalogue
            .find_by_class(&p.class_name)
            .map_err(internal_err)?,
        None => None,
    };

    let decision = state
        .thresholds
        .decide(top.map(|p| p.confidence).unwrap_or(0.0), matched_product.is_some());

    // Only show candidates that actually map to a real product, no
    // point offering the cashier something they can't ring up.
    let mut candidates = Vec::new();
    for p in &predictions {
        if let Some(product) = state.catalogue.find_by_class(&p.class_name).map_err(internal_err)? {
            candidates.push(json!({
                "sku": product.sku,
                "name": product.name,
                "unit_price": product.unit_price,
                "confidence": p.confidence,
            }));
        }
    }

    let latency_ms = start.elapsed().as_millis() as i64;
    let event_id = Uuid::new_v4().to_string();

    state
        .catalogue
        .log_event(
            &event_id,
            &image_hash,
            top.map(|p| p.class_name.as_str()),
            top.map(|p| p.confidence),
            decision.as_str(),
            matched_product.as_ref().map(|p| p.sku.as_str()),
            latency_ms,
        )
        .map_err(internal_err)?;

    tracing::info!(
        event_id,
        decision = decision.as_str(),
        latency_ms,
        "recognition event"
    );

    // Jev runs alongside the decision above, purely for comparison, it
    // never changes what the cashier actually sees. Any failure or
    // timeout here is swallowed, not surfaced, a slow/down Jev should
    // never hold up a sale.
    let jev_result = if state.jev.enabled && !candidates.is_empty() {
        let jev_candidates: Vec<jev::JevCandidate> = predictions
            .iter()
            .filter_map(|p| {
                state
                    .catalogue
                    .find_by_class(&p.class_name)
                    .ok()
                    .flatten()
                    .map(|product| jev::JevCandidate {
                        sku: product.sku,
                        name: product.name,
                        imagenet_class: p.class_name.clone(),
                        confidence: p.confidence,
                    })
            })
            .collect();
        let jev_config_key = state.jev.api_key.clone();
        let jev_model = state.jev.model.clone();
        let jev_enabled = state.jev.enabled;
        tokio::task::spawn_blocking(move || {
            let cfg = jev::JevConfig {
                enabled: jev_enabled,
                api_key: jev_config_key,
                model: jev_model,
            };
            jev::decide(&cfg, &jev_candidates)
        })
        .await
        .ok()
    } else {
        None
    };

    if let Some(result) = &jev_result {
        match result {
            Ok(outcome) => {
                let _ = state.catalogue.log_jev_comparison(
                    &event_id,
                    outcome.chosen_sku.as_deref(),
                    Some(outcome.confidence),
                    Some(outcome.latency_ms),
                    None,
                );
                tracing::info!(
                    event_id,
                    jev_sku = outcome.chosen_sku.as_deref().unwrap_or("none_confident"),
                    jev_confidence = outcome.confidence,
                    jev_latency_ms = outcome.latency_ms,
                    deterministic_decision = decision.as_str(),
                    "jev comparison"
                );
            }
            Err(e) => {
                let _ = state
                    .catalogue
                    .log_jev_comparison(&event_id, None, None, None, Some(&e.to_string()));
                tracing::warn!(event_id, error = %e, "jev call failed, deterministic decision unaffected");
            }
        }
    }

    let jev_json = jev_result.and_then(|r| r.ok()).map(|o| {
        json!({
            "chosen_sku": o.chosen_sku,
            "confidence": o.confidence,
            "latency_ms": o.latency_ms,
        })
    });

    Ok(Json(json!({
        "event_id": event_id,
        "decision": decision.as_str(),
        "auto_accepted": matched_product.as_ref().filter(|_| decision == decision::Decision::AutoAccept),
        "candidates": candidates,
        "raw_predictions": predictions,
        "latency_ms": latency_ms,
        "jev": jev_json,
    })))
}

#[derive(Deserialize)]
struct SearchQuery {
    q: Option<String>,
}

/// Manual product search/selection, used when recognition isn't confident.
async fn list_products(
    State(state): State<Arc<AppState>>,
    Query(q): Query<SearchQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let products = match &q.q {
        Some(query) if !query.is_empty() => state.catalogue.search(query),
        _ => state.catalogue.all_products(),
    }
    .map_err(internal_err)?;
    Ok(Json(json!({ "products": products })))
}

#[derive(Deserialize)]
struct ConfirmRequest {
    event_id: String,
    /// The SKU the cashier actually confirmed, whether or not it
    /// matches the model's top prediction, they can always override.
    sku: String,
    /// If the "save for training" checkbox was on, the same base64
    /// frame sent to /recognize, re-sent here now that we know its
    /// correct label. Saved to disk under training_data/<sku>/, never
    /// uploaded anywhere, purely local.
    image_base64: Option<String>,
}

/// Cashier's final word on a recognition event. Updates final_sku so
/// we can tell later whether the model was right, corrected, or
/// overridden entirely. Optionally saves the image as labeled training
/// data for a future fine-tuning pass.
async fn confirm(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ConfirmRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let product = state
        .catalogue
        .find_by_sku(&payload.sku)
        .map_err(internal_err)?
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("unknown SKU: {}", payload.sku) })),
            )
        })?;

    state
        .catalogue
        .update_final_sku(&payload.event_id, &payload.sku)
        .map_err(internal_err)?;

    tracing::info!(
        event_id = payload.event_id,
        sku = payload.sku,
        "cashier confirmed product"
    );

    let mut saved_training_image = false;
    if let Some(b64) = &payload.image_base64 {
        match save_training_image(&payload.sku, &payload.event_id, b64) {
            Ok(_) => {
                saved_training_image = true;
                tracing::info!(sku = payload.sku, "saved training image");
            }
            Err(e) => {
                // A failed save shouldn't fail the confirm itself, the
                // sale/confirmation already succeeded, this is just
                // bookkeeping for later.
                tracing::warn!(error = %e, "failed to save training image");
            }
        }
    }

    Ok(Json(json!({
        "confirmed": product,
        "saved_training_image": saved_training_image,
    })))
}

fn save_training_image(sku: &str, event_id: &str, image_base64: &str) -> Result<(), String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(image_base64)
        .map_err(|e| e.to_string())?;

    let dir = std::path::Path::new("training_data").join(sku);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let path = dir.join(format!("{event_id}.jpg"));
    std::fs::write(path, bytes).map_err(|e| e.to_string())?;

    Ok(())
}

/// Benchmark/test-result summary built from the logged events, plain
/// markdown so it can be pasted straight into the README.
async fn stats_md(
    State(state): State<Arc<AppState>>,
) -> Result<
    ([(axum::http::header::HeaderName, &'static str); 1], String),
    (StatusCode, Json<serde_json::Value>),
> {
    let md = state.catalogue.stats_markdown().map_err(internal_err)?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        md,
    ))
}

fn internal_err(e: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": e })),
    )
}
