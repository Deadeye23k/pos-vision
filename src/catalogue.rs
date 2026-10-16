// Product catalogue + recognition event logging (SQLite). Kept
// separate from vision.rs and decision.rs so each piece can change
// independently.

use rusqlite::Connection;
use serde::Serialize;
use std::sync::Mutex;

pub struct Catalogue {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Product {
    pub sku: String,
    pub name: String,
    pub unit_price: f64,
    /// The ImageNet class name this product is recognized as.
    /// A product could map to more than one class in principle
    /// (e.g. two apple varieties), so this stays a simple 1:1 for
    /// the M3 prototype rather than a join table.
    pub imagenet_class: String,
}

impl Catalogue {
    /// Opens (or creates) the SQLite file at `path`, creates tables if
    /// missing, and seeds the product catalogue if it's empty.
    pub fn open(path: &str) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| e.to_string())?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS products (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                unit_price REAL NOT NULL,
                imagenet_class TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recognition_events (
                id TEXT PRIMARY KEY,
                image_hash TEXT NOT NULL,
                predicted_class TEXT,
                predicted_confidence REAL,
                decision TEXT NOT NULL,
                final_sku TEXT,
                latency_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                jev_sku TEXT,
                jev_confidence REAL,
                jev_latency_ms INTEGER,
                jev_error TEXT,
                confirmed_at TEXT
            );
            ",
        )
        .map_err(|e| e.to_string())?;

        // Migration for DBs created before M4 added the jev_* columns.
        // ALTER TABLE ADD COLUMN fails harmlessly if the column already
        // exists, that failure is expected and ignored on purpose.
        for stmt in [
            "ALTER TABLE recognition_events ADD COLUMN jev_sku TEXT",
            "ALTER TABLE recognition_events ADD COLUMN jev_confidence REAL",
            "ALTER TABLE recognition_events ADD COLUMN jev_latency_ms INTEGER",
            "ALTER TABLE recognition_events ADD COLUMN jev_error TEXT",
            "ALTER TABLE recognition_events ADD COLUMN confirmed_at TEXT",
        ] {
            let _ = conn.execute(stmt, []);
        }

        let catalogue = Catalogue {
            conn: Mutex::new(conn),
        };
        catalogue.seed_if_empty()?;
        Ok(catalogue)
    }

    fn seed_if_empty(&self) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM products", [], |row| row.get(0))
            .map_err(|e| e.to_string())?;
        if count > 0 {
            return Ok(());
        }

        // Seed catalogue, 10 produce classes. These are all
        // real ImageNet-1k class names (see models/imagenet_classes.txt),
        // so recognition can map to them without any fine-tuning yet.
        // Prices are placeholders in MYR/kg or MYR/unit for the demo.
        let seed: &[(&str, &str, f64, &str)] = &[
            ("PRD-001", "Banana", 4.50, "banana"),
            ("PRD-002", "Orange", 6.90, "orange"),
            ("PRD-003", "Lemon", 8.20, "lemon"),
            ("PRD-004", "Broccoli", 9.50, "broccoli"),
            ("PRD-005", "Cauliflower", 7.80, "cauliflower"),
            ("PRD-006", "Zucchini", 6.50, "zucchini"),
            ("PRD-007", "Cucumber", 3.90, "cucumber"),
            ("PRD-008", "Bell Pepper", 10.90, "bell pepper"),
            ("PRD-009", "Mushroom", 12.50, "mushroom"),
            ("PRD-010", "Strawberry", 15.90, "strawberry"),
        ];

        for (sku, name, price, class) in seed {
            conn.execute(
                "INSERT INTO products (sku, name, unit_price, imagenet_class) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![sku, name, price, class],
            )
            .map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Maps a predicted ImageNet class name to a catalogue product,
    /// if one exists. Returns None for classes with no matching product
    /// (e.g. "digital clock"), which the caller should treat as a miss.
    pub fn find_by_class(&self, imagenet_class: &str) -> Result<Option<Product>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let result = conn.query_row(
            "SELECT sku, name, unit_price, imagenet_class FROM products WHERE imagenet_class = ?1",
            [imagenet_class],
            |row| {
                Ok(Product {
                    sku: row.get(0)?,
                    name: row.get(1)?,
                    unit_price: row.get(2)?,
                    imagenet_class: row.get(3)?,
                })
            },
        );
        match result {
            Ok(p) => Ok(Some(p)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Used by the manual fallback for product search/selection.
    /// A simple substring match is enough for a 5-10 item demo catalogue;
    /// this is the obvious place to add a proper search index later if
    /// the catalogue grows.
    pub fn search(&self, query: &str) -> Result<Vec<Product>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let like = format!("%{}%", query.to_lowercase());
        let mut stmt = conn
            .prepare(
                "SELECT sku, name, unit_price, imagenet_class FROM products
                 WHERE LOWER(name) LIKE ?1 ORDER BY name",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([&like], |row| {
                Ok(Product {
                    sku: row.get(0)?,
                    name: row.get(1)?,
                    unit_price: row.get(2)?,
                    imagenet_class: row.get(3)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
    }

    pub fn all_products(&self) -> Result<Vec<Product>, String> {
        self.search("")
    }

    pub fn find_by_sku(&self, sku: &str) -> Result<Option<Product>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let result = conn.query_row(
            "SELECT sku, name, unit_price, imagenet_class FROM products WHERE sku = ?1",
            [sku],
            |row| {
                Ok(Product {
                    sku: row.get(0)?,
                    name: row.get(1)?,
                    unit_price: row.get(2)?,
                    imagenet_class: row.get(3)?,
                })
            },
        );
        match result {
            Ok(p) => Ok(Some(p)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Records a recognition event, from the initial prediction
    /// through to the cashier's final selection (which may differ from
    /// the model's prediction, that's the whole point of logging it).
    #[allow(clippy::too_many_arguments)]
    pub fn log_event(
        &self,
        id: &str,
        image_hash: &str,
        predicted_class: Option<&str>,
        predicted_confidence: Option<f32>,
        decision: &str,
        final_sku: Option<&str>,
        latency_ms: i64,
    ) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT INTO recognition_events
                (id, image_hash, predicted_class, predicted_confidence, decision, final_sku, latency_ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                id,
                image_hash,
                predicted_class,
                predicted_confidence,
                decision,
                final_sku,
                latency_ms,
                chrono::Utc::now().to_rfc3339(),
            ],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// M4: records Jev's answer (or error) against an already-logged
    /// event, for offline comparison against the deterministic decision
    /// and against what the cashier actually confirmed. Called after
    /// the main event row already exists, so this is always an UPDATE.
    pub fn log_jev_comparison(
        &self,
        event_id: &str,
        jev_sku: Option<&str>,
        jev_confidence: Option<f32>,
        jev_latency_ms: Option<i64>,
        jev_error: Option<&str>,
    ) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE recognition_events
             SET jev_sku = ?1, jev_confidence = ?2, jev_latency_ms = ?3, jev_error = ?4
             WHERE id = ?5",
            rusqlite::params![jev_sku, jev_confidence, jev_latency_ms, jev_error, event_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Cashier confirmed (or corrected) the product for an existing
    /// recognition event. Called from /confirm, after the event was
    /// already logged by /recognize.
    pub fn update_final_sku(&self, event_id: &str, sku: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let rows = conn
            .execute(
                "UPDATE recognition_events SET final_sku = ?1, confirmed_at = ?3 WHERE id = ?2",
                rusqlite::params![sku, event_id, chrono::Utc::now().to_rfc3339()],
            )
            .map_err(|e| e.to_string())?;
        if rows == 0 {
            return Err(format!("no recognition event found with id {event_id}"));
        }
        Ok(())
    }
}

// ---- Benchmark / test-result stats --------------------------------
// Everything below is computed from the recognition_events table, so
// the numbers in the README come from real test runs, not estimates.

struct EventRow {
    decision: String,
    predicted_confidence: Option<f64>,
    top_sku: Option<String>,
    final_sku: Option<String>,
    confirmed: bool,
    latency_ms: i64,
    jev_sku: Option<String>,
    jev_latency_ms: Option<i64>,
    jev_error: Option<String>,
}

fn percentile(sorted: &[i64], p: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

impl Catalogue {
    /// Returns a ready-to-paste markdown summary of all logged events.
    pub fn stats_markdown(&self) -> Result<String, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;

        let mut stmt = conn
            .prepare(
                "SELECT e.decision, e.predicted_confidence, p.sku, e.final_sku,
                        e.confirmed_at IS NOT NULL, e.latency_ms,
                        e.jev_sku, e.jev_latency_ms, e.jev_error
                 FROM recognition_events e
                 LEFT JOIN products p ON p.imagenet_class = e.predicted_class
                 ORDER BY e.created_at",
            )
            .map_err(|e| e.to_string())?;

        let rows: Vec<EventRow> = stmt
            .query_map([], |r| {
                Ok(EventRow {
                    decision: r.get(0)?,
                    predicted_confidence: r.get(1)?,
                    top_sku: r.get(2)?,
                    final_sku: r.get(3)?,
                    confirmed: r.get(4)?,
                    latency_ms: r.get(5)?,
                    jev_sku: r.get(6)?,
                    jev_latency_ms: r.get(7)?,
                    jev_error: r.get(8)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        let names: std::collections::HashMap<String, String> = {
            let mut s = conn
                .prepare("SELECT sku, name FROM products ORDER BY sku")
                .map_err(|e| e.to_string())?;
            let it = s
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
                .map_err(|e| e.to_string())?;
            it.collect::<Result<_, _>>().map_err(|e| e.to_string())?
        };

        let total = rows.len();
        if total == 0 {
            return Ok("No recognition events logged yet. Run some tests first.\n".to_string());
        }

        let mut out = String::new();
        out.push_str(&format!("## Test results ({total} recognition events)\n\n"));

        // Latency
        let mut lat: Vec<i64> = rows.iter().map(|r| r.latency_ms).collect();
        lat.sort();
        out.push_str("### Latency (server-side, /recognize)\n\n");
        out.push_str("| p50 | p95 | max | runs |\n|---|---|---|---|\n");
        out.push_str(&format!(
            "| {} ms | {} ms | {} ms | {} |\n\n",
            percentile(&lat, 0.5),
            percentile(&lat, 0.95),
            lat.last().copied().unwrap_or(0),
            total
        ));

        // Decisions
        let count = |d: &str| rows.iter().filter(|r| r.decision == d).count();
        out.push_str("### Decision routing\n\n");
        out.push_str("| AUTO_ACCEPT | REVIEW | MANUAL |\n|---|---|---|\n");
        out.push_str(&format!(
            "| {} | {} | {} |\n\n",
            count("AUTO_ACCEPT"),
            count("REVIEW"),
            count("MANUAL")
        ));

        // Per-product accuracy (confirmed events only)
        let confirmed: Vec<&EventRow> = rows.iter().filter(|r| r.confirmed).collect();
        out.push_str("### Recognition accuracy by product\n\n");
        out.push_str(&format!(
            "Based on {} cashier-confirmed events. \"Top-1 correct\" means the model's first guess mapped to the product the tester confirmed.\n\n",
            confirmed.len()
        ));
        out.push_str("| Product | Tests | Top-1 correct | Avg top-1 confidence |\n|---|---|---|---|\n");
        let mut skus: Vec<&String> = names.keys().collect();
        skus.sort();
        for sku in skus {
            let evs: Vec<&&EventRow> = confirmed
                .iter()
                .filter(|r| r.final_sku.as_deref() == Some(sku.as_str()))
                .collect();
            if evs.is_empty() {
                continue;
            }
            let correct = evs
                .iter()
                .filter(|r| r.top_sku.as_deref() == Some(sku.as_str()))
                .count();
            let confs: Vec<f64> = evs.iter().filter_map(|r| r.predicted_confidence).collect();
            let avg = if confs.is_empty() {
                0.0
            } else {
                confs.iter().sum::<f64>() / confs.len() as f64
            };
            out.push_str(&format!(
                "| {} | {} | {}/{} | {:.1}% |\n",
                names.get(sku).cloned().unwrap_or_else(|| sku.clone()),
                evs.len(),
                correct,
                evs.len(),
                avg * 100.0
            ));
        }
        let all_correct = confirmed
            .iter()
            .filter(|r| r.top_sku.is_some() && r.top_sku == r.final_sku)
            .count();
        out.push_str(&format!(
            "\nOverall top-1 accuracy: {}/{} confirmed events.\n\n",
            all_correct,
            confirmed.len()
        ));

        // Jev comparison
        let jev_ok: Vec<&EventRow> = rows.iter().filter(|r| r.jev_latency_ms.is_some()).collect();
        let jev_err = rows.iter().filter(|r| r.jev_error.is_some()).count();
        out.push_str("### Jev vs deterministic thresholds\n\n");
        if jev_ok.is_empty() && jev_err == 0 {
            out.push_str("Jev was not enabled (or never called) during these runs.\n");
        } else {
            let mut jl: Vec<i64> = jev_ok.iter().filter_map(|r| r.jev_latency_ms).collect();
            jl.sort();
            out.push_str(&format!(
                "Jev calls: {} succeeded, {} failed. Jev latency: p50 {} ms, p95 {} ms.\n\n",
                jev_ok.len(),
                jev_err,
                percentile(&jl, 0.5),
                percentile(&jl, 0.95)
            ));
            let both: Vec<&&EventRow> = jev_ok.iter().filter(|r| r.confirmed).collect();
            if !both.is_empty() {
                let det = both
                    .iter()
                    .filter(|r| r.top_sku.is_some() && r.top_sku == r.final_sku)
                    .count();
                let jev = both
                    .iter()
                    .filter(|r| r.jev_sku.is_some() && r.jev_sku == r.final_sku)
                    .count();
                let none = both.iter().filter(|r| r.jev_sku.is_none()).count();
                out.push_str("| Method | Matched confirmed product |\n|---|---|\n");
                out.push_str(&format!("| Deterministic top-1 | {}/{} |\n", det, both.len()));
                out.push_str(&format!("| Jev choice | {}/{} |\n", jev, both.len()));
                out.push_str(&format!(
                    "\nJev answered \"none confident\" on {} of {} events.\n",
                    none,
                    both.len()
                ));
            }
        }

        Ok(out)
    }
}
