use crate::tensors::Tensor;
use js_sys::{Function, Object};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Executes the provided function fn and after it is executed, cleans up all intermediate
    /// tensors allocated by fn except those returned by fn. fn must not return a Promise (async
    /// functions not allowed). The returned result can be a complex object.
    pub fn tidy(name_or_fn: &JsValue, fun: Option<&Function>) -> JsValue;

    /// Disposes any Tensors found within the provided object.
    pub fn dispose(container: &JsValue);

    /// Keeps a Tensor generated inside a tidy() from being disposed automatically.
    pub fn keep(result: &Tensor) -> Tensor;

    /// Returns memory info at the current time in the program.
    pub fn memory() -> Object;
}
