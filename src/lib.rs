use js_sys::{Array, Promise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A Tensor object represents an immutable, multidimensional array of numbers that has a
    /// shape and a data type.
    #[wasm_bindgen(extends = Promise)]
    pub type Tensor;

    /// Prints information about the Tensor including its data.
    #[wasm_bindgen(method)]
    pub fn print(this: &Tensor, verbose: bool);

    /// Creates rank-0 Tensor (scalar) with the provided value and dtype.
    pub fn scalar(value: &JsValue, dtype: Option<&str>) -> Tensor;

    /// Creates a Tensor with the provided values, shape and dtype.
    pub fn tensor(values: &Array, shape: Option<&Array>, dtype: Option<&str>) -> Tensor;
}
