use js_sys::Array;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A Tensor object represents an immutable, multidimensional array of numbers that has a
    /// shape and a data type.
    pub type Tensor;

    /// Prints information about the Tensor including its data.
    #[wasm_bindgen(method)]
    pub fn print(this: &Tensor, verbose: bool);

    /// Creates rank-0 Tensor (scalar) with the provided value and dtype.
    pub fn scalar(value: &JsValue, dtype: Option<&str>) -> Tensor;

    /// Creates a Tensor with the provided values, shape and dtype.
    pub fn tensor(values: &Array, shape: Option<&Array>, dtype: Option<&str>) -> Tensor;
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A mutable Tensor, useful for persisting state, e.g. for training.
    #[wasm_bindgen(extends = Tensor)]
    pub type Variable;

    /// Assign a new Tensor to this variable. The new Tensor must have the same shape and dtype as
    /// the old Tensor.
    #[wasm_bindgen(method)]
    pub fn assign(this: &Variable, value: &Tensor);

    #[wasm_bindgen(constructor)]
    pub fn new(that: &Tensor) -> Variable;
}
