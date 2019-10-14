use crate::tensors::Tensor;
use js_sys::{Function, Object, Promise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Disposes any Tensors found within the provided object.
    pub fn dispose(container: &JsValue);

    /// Keeps a Tensor generated inside a tidy() from being disposed automatically.
    pub fn keep(result: &Tensor) -> Tensor;

    /// Returns memory info at the current time in the program.
    pub fn memory() -> Object;

    /// Returns a promise that resolve when a requestAnimationFrame has completed.
    #[wasm_bindgen(js_name = "nextFrame")]
    #[must_use]
    pub fn next_frame() -> Promise;

    /// Executes the provided function f() and returns a promise that resolves with information
    /// about the function's memory use.
    #[must_use]
    pub fn profile(f: &Function) -> Promise;

    /// Executes the provided function fn and after it is executed, cleans up all intermediate
    /// tensors allocated by fn except those returned by fn. fn must not return a Promise (async
    /// functions not allowed). The returned result can be a complex object.
    pub fn tidy(name_or_fn: &JsValue, fun: Option<&Function>) -> JsValue;

    /// Executes f() and returns a promise that resolves with timing information.
    #[must_use]
    pub fn time(f: &Function) -> Promise;
}
