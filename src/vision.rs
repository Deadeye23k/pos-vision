// Local vision recognition.
//
// Loads the MobileNetV2 ONNX model once at startup, and exposes
// `recognize(image_bytes) -> Vec<Prediction>` for the /recognize route
// in main.rs. Kept separate so the catalogue/Jev logic can build on
// top of this without needing to touch it.

use image::imageops::FilterType;
use ndarray::Array4;
use once_cell::sync::OnceCell;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::sync::Mutex;

const INPUT_SIZE: u32 = 224;
// Standard ImageNet normalization, matches what the model was trained with.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

static SESSION: OnceCell<Mutex<Session>> = OnceCell::new();
static CLASS_NAMES: OnceCell<Vec<String>> = OnceCell::new();

#[derive(Debug, serde::Serialize)]
pub struct Prediction {
    pub class_name: String,
    pub confidence: f32,
}

#[derive(Debug)]
pub enum VisionError {
    ModelNotLoaded,
    DecodeImage(String),
    Inference(String),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisionError::ModelNotLoaded => write!(f, "vision model not loaded"),
            VisionError::DecodeImage(e) => write!(f, "could not decode image: {e}"),
            VisionError::Inference(e) => write!(f, "inference failed: {e}"),
        }
    }
}

/// Loads the ONNX model and class labels. Call once at server startup.
/// Returns an error rather than panicking, since a missing model file
/// should be a clear startup failure, not a crash mid-request.
pub fn init(model_path: &str, labels_path: &str) -> Result<(), String> {
    let session = Session::builder()
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?
        .commit_from_file(model_path)
        .map_err(|e| format!("loading {model_path}: {e}"))?;

    let labels = std::fs::read_to_string(labels_path)
        .map_err(|e| format!("loading {labels_path}: {e}"))?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>();

    SESSION
        .set(Mutex::new(session))
        .map_err(|_| "vision session already initialized".to_string())?;
    CLASS_NAMES
        .set(labels)
        .map_err(|_| "class labels already initialized".to_string())?;

    Ok(())
}

/// Runs inference on a raw image (any format `image` crate supports:
/// JPEG/PNG/WebP), returns the top-k predictions sorted by confidence.
pub fn recognize(image_bytes: &[u8], top_k: usize) -> Result<Vec<Prediction>, VisionError> {
    let session_lock = SESSION.get().ok_or(VisionError::ModelNotLoaded)?;
    let class_names = CLASS_NAMES.get().ok_or(VisionError::ModelNotLoaded)?;

    let img = image::load_from_memory(image_bytes)
        .map_err(|e| VisionError::DecodeImage(e.to_string()))?
        .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
        .to_rgb8();

    // HWC u8 -> CHW f32, normalized per-channel.
    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for (x, y, pixel) in img.enumerate_pixels() {
        for c in 0..3 {
            let v = pixel[c] as f32 / 255.0;
            input[[0, c, y as usize, x as usize]] = (v - MEAN[c]) / STD[c];
        }
    }

    let session = session_lock
        .lock()
        .map_err(|e| VisionError::Inference(e.to_string()))?;

    let input_tensor =
        Tensor::from_array(input).map_err(|e| VisionError::Inference(e.to_string()))?;
    let inputs =
        ort::inputs![input_tensor].map_err(|e| VisionError::Inference(e.to_string()))?;
    let outputs = session
        .run(inputs)
        .map_err(|e| VisionError::Inference(e.to_string()))?;

    let logits = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| VisionError::Inference(e.to_string()))?;

    let logits_vec: Vec<f32> = logits.iter().copied().collect();
    let probs = softmax(&logits_vec);

    let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Ok(ranked
        .into_iter()
        .take(top_k)
        .map(|(idx, conf)| Prediction {
            class_name: class_names
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("class_{idx}")),
            confidence: conf,
        })
        .collect())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}
