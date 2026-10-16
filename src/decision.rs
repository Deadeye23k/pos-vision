// Turns a confidence score into AUTO_ACCEPT / REVIEW / MANUAL. High
// confidence auto-accepts, medium shows candidates, low goes to manual
// search. Thresholds are configurable via env vars instead of hard-coded.
//
// No dependency on vision.rs, catalogue.rs, or the web layer, so it's
// easy to unit test on its own.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Decision {
    AutoAccept,
    Review,
    Manual,
}

impl Decision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Decision::AutoAccept => "AUTO_ACCEPT",
            Decision::Review => "REVIEW",
            Decision::Manual => "MANUAL",
        }
    }
}

pub struct Thresholds {
    /// Confidence at or above this: auto-accept the top prediction.
    pub auto_accept: f32,
    /// Confidence at or above this (but below auto_accept): show
    /// top-3 for the cashier to pick from. Below this: manual search.
    pub review: f32,
}

impl Thresholds {
    /// Reads POS_AUTO_ACCEPT_THRESHOLD / POS_REVIEW_THRESHOLD from the
    /// environment, falling back to defaults if unset or unparseable.
    /// Defaults lean conservative on purpose: better to ask the cashier
    /// too often than to silently mis-ring an item.
    pub fn from_env() -> Self {
        let auto_accept = std::env::var("POS_AUTO_ACCEPT_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.70);
        let review = std::env::var("POS_REVIEW_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.30);
        Thresholds {
            auto_accept,
            review,
        }
    }

    /// `matched_catalogue` is whether the top prediction's class maps
    /// to a real product. A high-confidence guess for something we
    /// don't sell (e.g. "digital clock") should never auto-accept,
    /// there's nothing sensible to accept into.
    pub fn decide(&self, top_confidence: f32, matched_catalogue: bool) -> Decision {
        if !matched_catalogue {
            return Decision::Manual;
        }
        if top_confidence >= self.auto_accept {
            Decision::AutoAccept
        } else if top_confidence >= self.review {
            Decision::Review
        } else {
            Decision::Manual
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> Thresholds {
        Thresholds {
            auto_accept: 0.70,
            review: 0.30,
        }
    }

    #[test]
    fn high_confidence_matched_class_auto_accepts() {
        assert_eq!(thresholds().decide(0.95, true), Decision::AutoAccept);
    }

    #[test]
    fn boundary_confidence_auto_accepts() {
        assert_eq!(thresholds().decide(0.70, true), Decision::AutoAccept);
    }

    #[test]
    fn medium_confidence_goes_to_review() {
        assert_eq!(thresholds().decide(0.50, true), Decision::Review);
    }

    #[test]
    fn low_confidence_goes_manual() {
        assert_eq!(thresholds().decide(0.05, true), Decision::Manual);
    }

    #[test]
    fn high_confidence_but_unmatched_class_still_goes_manual() {
        // e.g. model is 99% sure it's a "digital clock", which isn't
        // a product we sell, this must never auto-accept.
        assert_eq!(thresholds().decide(0.99, false), Decision::Manual);
    }

    #[test]
    fn env_overrides_are_read_correctly() {
        std::env::set_var("POS_AUTO_ACCEPT_THRESHOLD", "0.9");
        std::env::set_var("POS_REVIEW_THRESHOLD", "0.4");
        let t = Thresholds::from_env();
        assert_eq!(t.auto_accept, 0.9);
        assert_eq!(t.review, 0.4);
        std::env::remove_var("POS_AUTO_ACCEPT_THRESHOLD");
        std::env::remove_var("POS_REVIEW_THRESHOLD");
    }
}
