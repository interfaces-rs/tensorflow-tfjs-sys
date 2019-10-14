use crate::tensors::Tensor;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Binary accuracy metric function.
    #[wasm_bindgen(js_name = "binaryAccuracy")]
    pub fn binary_accuracy(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Binary crossentropy metric function.
    #[wasm_bindgen(js_name = "binaryCrossentropy")]
    pub fn binary_crossentropy(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Categorical accuracy metric function.
    #[wasm_bindgen(js_name = "categoricalAccuracy")]
    pub fn categorical_accuracy(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Categorical crossentropy between an output tensor and a target tensor.
    #[wasm_bindgen(js_name = "categoricalCrossentropy")]
    pub fn categorical_crossentropy(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Loss or metric function: Cosine proximity.
    #[wasm_bindgen(js_name = "cosineProximity")]
    pub fn cosine_proximity(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Loss or metric function: Mean absolute error.
    #[wasm_bindgen(js_name = "meanAbsoluteError")]
    pub fn mean_absolute_error(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Loss or metric function: Mean absolute percentage error.
    #[wasm_bindgen(js_name = "meanAbsolutePercentageError")]
    pub fn mean_absolute_percentage_error(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Loss or metric function: Mean squared error.
    #[wasm_bindgen(js_name = "meanSquaredError")]
    pub fn mean_squared_error(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Computes the precision of the predictions with respect to the labels.
    pub fn precision(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Computes the recall of the predictions with respect to the labels.
    pub fn recall(y_true: &Tensor, y_pred: &Tensor) -> Tensor;

    /// Sparse categorical accuracy metric function.
    #[wasm_bindgen(js_name = "sparseCategoricalAccuracy")]
    pub fn sparse_categorical_accuracy(y_true: &Tensor, y_pred: &Tensor) -> Tensor;
}
