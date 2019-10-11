use js_sys::{Object, Promise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A GraphModel is a directed, acyclic graph of built from SavedModel GraphDef and allows
    /// inference exeuction.
    pub type GraphModel;

    /// Execute the inference for the input tensors.
    #[wasm_bindgen(method)]
    pub fn predict(this: &GraphModel, inputs: &JsValue, config: Option<&Object>) -> JsValue;

    /// Executes inference for the model for given input tensors.
    #[wasm_bindgen(method)]
    pub fn execute(this: &GraphModel, inputs: &JsValue, outputs: Option<&Object>) -> JsValue;

    /// Executes inference for the model for given input tensors in async fashion, use this method
    /// when your model contains control flow ops.
    #[wasm_bindgen(method, js_name = "executeAsync")]
    pub fn execute_async(this: &GraphModel, inputs: &JsValue, outputs: Option<&Object>) -> Promise;

    /// Releases the memory used by the weight tensors.
    #[wasm_bindgen(method)]
    pub fn dispose(this: &GraphModel);
}
