#![deny(clippy::all)]

use napi_derive::napi;
use napi::bindgen_prelude::*; // For BigInt, Buffer, etc.

#[napi]
pub struct MoguImageProcessor {
    inner: mogu::ImageProcessor,
}

#[napi]
impl MoguImageProcessor {
    #[napi(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            inner: mogu::ImageProcessor::new(width, height),
        }
    }

    #[napi(factory)]
    pub fn with_imagenet_normalization(width: u32, height: u32) -> Self {
        Self {
            inner: mogu::ImageProcessor::with_imagenet_normalization(width, height),
        }
    }

    #[napi]
    pub fn preprocess(&self, buffer: Buffer) -> Result<MoguTensor> {
        let img = image::load_from_memory(&buffer)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load image: {}", e)))?;
        
        let tensor = self.inner.preprocess(&img);
        Ok(MoguTensor(tensor))
    }
}

// Wrapper struct for array4<f32>
// Since we can't easily expose generic types or complex structs directly to JS without conversion,
// we wrap it in a struct that JS holds as an opaque object (or we could expose data if needed).
// For now, it's just a handle to pass between processor and detector.
#[napi]
pub struct MoguTensor(pub(crate) ndarray::Array4<f32>);

#[napi]
pub struct MoguFoodDetector {
    inner: mogu::FoodDetector,
}

#[napi(object)]
pub struct PredictionResult {
    pub index: u32,
    pub probability: f64,
}

#[napi(object)]
pub struct IsFoodResult {
    pub is_food: bool,
    pub score: f64,
}

#[napi]
impl MoguFoodDetector {
    #[napi(constructor)]
    pub fn new(model_path: String) -> Result<Self> {
        let inner = mogu::FoodDetector::new(model_path)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load model: {}", e)))?;
        Ok(Self { inner })
    }

    #[napi]
    pub fn predict(&mut self, tensor: &MoguTensor) -> Result<Vec<PredictionResult>> {
        // Clone the array because predict consumes it in original code? 
        // Checking original code: predict takes Array4<f32>.
        // So we need to clone it.
        let input = tensor.0.clone();
        
        let results = self.inner.predict(input)
            .map_err(|e| napi::Error::from_reason(format!("Prediction failed: {}", e)))?;
        
        Ok(results.into_iter().map(|(idx, prob)| PredictionResult {
            index: idx as u32,
            probability: prob as f64,
        }).collect())
    }

    #[napi]
    pub fn predict_is_food(&mut self, tensor: &MoguTensor) -> Result<IsFoodResult> {
        let input = tensor.0.clone();
        
        let (is_food, score) = self.inner.predict_is_food(input)
             .map_err(|e| napi::Error::from_reason(format!("Prediction failed: {}", e)))?;
             
        Ok(IsFoodResult {
            is_food,
            score: score as f64,
        })
    }
}
