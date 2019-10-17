mod browser;
mod callbacks;
mod constraints;
mod data;
mod initializers;
pub mod io;
mod layers;
mod metrics;
mod models;
mod performance;
mod tensors;
mod training;
mod util;

pub use layers::*;
pub use models::*;
pub use performance::*;
pub use tensors::*;
pub use training::*;

use js_sys::JsString;
use wasm_bindgen::prelude::*;

pub type DType = &'static str;

// FIXME
pub type OpMapper = js_sys::Object;

#[wasm_bindgen]
extern {
    pub type Activation;
}

#[wasm_bindgen]
extern {
    pub type Callbacks;

    /// Factory function for a Callback that stops training when a monitored quantity has stopped
    /// improving.
    #[wasm_bindgen(js_name = "earlyStopping")]
    pub fn early_stopping(this: &Callbacks, args: Option<&Object>) -> EarlyStopping;

    pub static callbacks: Callbacks;
}

#[wasm_bindgen]
extern {
    pub type DatasetContainer;
}

#[wasm_bindgen]
extern {
    pub type DisposeResult;

    /// Number of variables dispose in this dispose call.
    #[wasm_bindgen(method, getter, js_name = "numDisposedVariables")]
    pub fn num_disposed_variables(this: &DisposeResult) -> usize;

    /// Reference count after the dispose call.
    #[wasm_bindgen(method, getter, js_name = "refCountAfterDispose")]
    pub fn ref_count_after_dispose(this: &DisposeResult) -> usize;
}

#[wasm_bindgen]
extern {
    pub type EarlyStopping;
}

#[wasm_bindgen]
extern {
    pub type Initializer;
}

#[wasm_bindgen]
extern {
    pub type IOHandler;
}

#[wasm_bindgen]
extern {
    pub type Metrics;

    /// Binary accuracy metric function.
    pub fn binary_accuracy(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Binary crossentropy metric function.
    pub fn binary_crossentropy(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Categorical accuracy metric function.
    pub fn categorical_accuracy(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Categorical crossentropy between an output tensor and a target tensor.
    pub fn categorical_crossentropy(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Loss or metric function: Cosine proximity.
    pub fn cosine_proximity(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Loss or metric function: Mean absolute error.
    pub fn mean_absolute_error(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Loss or metric function: Mean absolute percentage error.
    pub fn mean_absolute_percentage_error(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Loss or metric function: Mean squared error.
    pub fn mean_squared_error(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Computes the precision of the predictions with respect to the labels.
    pub fn precision(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Computes the recall of the predictions with respect to the labels.
    pub fn recall(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Sparse categorical accuracy metric function.
    pub fn sparse_categorical_accuracy(this: &Metrics, y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    pub static metrics: Metrics;
}

#[wasm_bindgen]
extern {
    pub type Regularizer;
}

#[wasm_bindgen]
extern {
    /// Serializable defines the serialization contract.
    pub type Serializable;

    /// Return the class name for this class to use in serialization contexts.
    #[wasm_bindgen(method)]
    pub fn get_class_name(this: &Serializable) -> JsString;

    /// Return all the non-weight state needed to serialize this object.
    #[wasm_bindgen(method)]
    pub fn get_config(this: &Serializable) -> JsString;

// FIXME
// pub fn from_config(class, config);
}
