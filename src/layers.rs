use crate::{Constraint, DisposeResult, Initializer, Regularizer};
use js_sys::{Array, JsString, Object};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A Layer is a grouping of operations and weights that can be composed to create a
    /// LayersModel.
    pub type Layer;

    /// Builds or executes a `Layer's logic.
    #[wasm_bindgen(method)]
    pub fn apply(this: &Layer, inputs: &JsValue, kwargs: Option<&Object>) -> JsValue;

    /// Counts the total number of numbers (e.g., float32, int32) in the weights.
    #[wasm_bindgen(method, js_name = "countParams")]
    pub fn count_params(this: &Layer) -> usize;

    /// Creates the layer weights.
    #[wasm_bindgen(method)]
    pub fn build(this: &Layer, input_shape: &JsValue);

    /// Returns the current values of the weights of the layer.
    #[wasm_bindgen(method, js_name = "getWeights")]
    pub fn get_weights(this: &Layer, trainable_only: Option<bool>) -> Box<[JsValue]>;

    /// Sets the weights of the layer, from Tensors.
    #[wasm_bindgen(method, js_name = "setWeights")]
    pub fn set_weights(this: &Layer, weights: Box<[JsValue]>);

    /// Adds a weight variable to the layer.
    #[wasm_bindgen(method, js_name = "addWeights")]
    pub fn add_weight(this: &Layer, name: &JsString, shape: &Array, dtype: Option<&str>, initializer: Option<&Initializer>, regularizer: Option<&Regularizer>, trainable: Option<bool>, constraint: Option<&Constraint>);

    /// Add losses to the layer.
    #[wasm_bindgen(method, js_name = "addLoss")]
    pub fn add_loss(this: &Layer, losses: &JsValue);

    /// Computes the output shape of the layer.
    #[wasm_bindgen(method, js_name = "computeOutputShape")]
    pub fn compute_output_shape(this: &Layer, input_shape: &JsValue) -> JsValue;

    /// Returns the config of the layer.
    #[wasm_bindgen(method)]
    pub fn get_config(this: &Layer) -> Object;

    /// Attempt to dispose layer's weights.
    #[wasm_bindgen(method)]
    #[must_use]
    pub fn dispose(this: &Layer) -> DisposeResult;
}
