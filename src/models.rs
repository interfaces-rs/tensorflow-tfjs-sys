use js_sys::{Object, Promise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A GraphModel is a directed, acyclic graph of built from SavedModel GraphDef and allows
    /// inference exeuction.
    pub type GraphModel;

    /// Execute the inference for the input tensors.
    pub fn predict(inputs: &JsValue, config: Option<&Object>) -> JsValue;

    /// Executes inference for the model for given input tensors.
    pub fn execute(inputs: &JsValue, outputs: Option<&Object>) -> JsValue;

    /// Executes inference for the model for given input tensors in async fashion, use this method
    /// when your model contains control flow ops.
    pub fn execute_async(inputs: &JsValue, outputs: Option<&Object>) -> Promise;

    /// Releases the memory used by the weight tensors.
    pub fn dispose();
}
