// Jev integration, behind a feature flag, compared against the
// deterministic thresholds in decision.rs rather than replacing them.
//
// Jev (api.typesafe.ai) is a decision-primitive model: given a `state`
// description and typed `questions` (noul/choice/score), it returns
// calibrated answers instead of generated text. We use a `choice`
// question here: given the vision model's candidates, which one (if
// any) is the right product?
//
// Jev is a hosted, external service, so a slow or down Jev must never
// block a sale. Every call has a timeout, and every error path just
// returns an error here rather than panicking, decision.rs's call
// always wins regardless of what happens with Jev.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const JEV_BASE_URL: &str = "https://api.typesafe.ai/v1/decisions";
const JEV_TIMEOUT: Duration = Duration::from_millis(800);

pub struct JevConfig {
    pub enabled: bool,
    pub api_key: String,
    pub model: String,
}

impl JevConfig {
    /// Jev only activates if JEV_ENABLED=true AND a key is present.
    /// Missing either one quietly disables it, this is meant to be
    /// an opt-in experiment, not something that can accidentally
    /// start depending on an external service.
    pub fn from_env() -> Self {
        let enabled = std::env::var("JEV_ENABLED")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        let api_key = std::env::var("TYPESAFE_API_KEY").unwrap_or_default();
        let model = std::env::var("JEV_MODEL").unwrap_or_else(|_| "jev-latest".to_string());

        JevConfig {
            enabled: enabled && !api_key.is_empty(),
            api_key,
            model,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JevCandidate {
    pub sku: String,
    pub name: String,
    pub imagenet_class: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct JevOutcome {
    /// SKU Jev picked, or None if it chose "none_confident".
    pub chosen_sku: Option<String>,
    pub confidence: f32,
    pub latency_ms: i64,
    /// Full per-option probability breakdown, kept for comparison
    /// logging/analysis, not used in the decision itself.
    pub probabilities: serde_json::Value,
}

#[derive(Debug)]
pub enum JevError {
    Disabled,
    Network(String),
    BadResponse(String),
}

impl std::fmt::Display for JevError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JevError::Disabled => write!(f, "Jev not enabled"),
            JevError::Network(e) => write!(f, "Jev network error: {e}"),
            JevError::BadResponse(e) => write!(f, "Jev returned unexpected response: {e}"),
        }
    }
}

#[derive(Serialize)]
struct DecisionsRequest {
    model: String,
    state: String,
    questions: std::collections::HashMap<String, Question>,
}

#[derive(Serialize)]
struct Question {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: String,
    criteria: std::collections::HashMap<String, String>,
}

#[derive(Deserialize)]
struct DecisionsResponse {
    answers: std::collections::HashMap<String, Answer>,
}

#[derive(Deserialize)]
struct Answer {
    choice: Option<String>,
    confidence: Option<f32>,
    probabilities: Option<serde_json::Value>,
}

/// Asks Jev to pick the best-matching product from the candidates the
/// deterministic path already found (same candidates shown to the
/// cashier for REVIEW), plus a "none_confident" escape hatch mirroring
/// our own MANUAL fallback. This is a blocking HTTP call, so callers
/// in the async handler must run it via `spawn_blocking`.
pub fn decide(
    config: &JevConfig,
    candidates: &[JevCandidate],
) -> Result<JevOutcome, JevError> {
    if !config.enabled {
        return Err(JevError::Disabled);
    }

    let state = format!(
        "A point-of-sale camera captured an image of a produce item at checkout. \
         A local vision model proposed the following candidates, each with the \
         ImageNet class it was recognized as and the model's confidence: {}",
        candidates
            .iter()
            .map(|c| format!(
                "'{}' (recognized as '{}', {:.1}% confidence)",
                c.name,
                c.imagenet_class,
                c.confidence * 100.0
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );

    let mut criteria: std::collections::HashMap<String, String> = candidates
        .iter()
        .map(|c| {
            (
                c.sku.clone(),
                format!(
                    "The item is a {}, matching the '{}' recognition candidate",
                    c.name, c.imagenet_class
                ),
            )
        })
        .collect();
    criteria.insert(
        "none_confident".to_string(),
        "None of the candidates are a confident match; a human cashier should pick manually"
            .to_string(),
    );

    let mut questions = std::collections::HashMap::new();
    questions.insert(
        "best_match".to_string(),
        Question {
            kind: "choice",
            instructions: "Which product does the captured image most likely show, or is none of them a confident match?".to_string(),
            criteria,
        },
    );

    let request = DecisionsRequest {
        model: config.model.clone(),
        state,
        questions,
    };

    let start = Instant::now();
    let result = ureq::post(JEV_BASE_URL)
        .set("Authorization", &format!("Bearer {}", config.api_key))
        .set("Content-Type", "application/json")
        .timeout(JEV_TIMEOUT)
        .send_json(&request);

    let response = match result {
        Ok(r) => r,
        Err(e) => return Err(JevError::Network(e.to_string())),
    };

    let parsed: DecisionsResponse = response
        .into_json()
        .map_err(|e| JevError::BadResponse(e.to_string()))?;

    let latency_ms = start.elapsed().as_millis() as i64;

    let answer = parsed
        .answers
        .get("best_match")
        .ok_or_else(|| JevError::BadResponse("missing 'best_match' in response".to_string()))?;

    let chosen = answer.choice.as_deref().unwrap_or("none_confident");
    let chosen_sku = if chosen == "none_confident" {
        None
    } else {
        Some(chosen.to_string())
    };

    Ok(JevOutcome {
        chosen_sku,
        confidence: answer.confidence.unwrap_or(0.0),
        latency_ms,
        probabilities: answer.probabilities.clone().unwrap_or(serde_json::json!({})),
    })
}
