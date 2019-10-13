use crate::DType;
use js_sys::{Array, Float32Array};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// SymbolicTensor is a placeholder for a Tensor without any concrete value.
    pub type SymbolicTensor;
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A Tensor object represents an immutable, multidimensional array of numbers that has a
    /// shape and a data type.
    pub type Tensor;

    /// Prints information about the Tensor including its data.
    #[wasm_bindgen(method)]
    pub fn print(this: &Tensor, verbose: bool);
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A mutable object, similar to Tensor, that allows users to set values at locations before
    /// converting to an immutable Tensor.
    pub type TensorBuffer;

    /// Creates an empty TensorBuffer with the specified shape and dtype.
    ///
    /// The values are stored in CPU as TypedArray. Fill the buffer using buffer.set(), or by
    /// modifying directly buffer.values.
    ///
    /// When done, call buffer.toTensor() to get an immutable Tensor with those values.
    pub fn buffer(shape: &Array, dtype: Option<DType>, values: Option<&Float32Array>) -> TensorBuffer;

    /// Sets a value in the buffer at a given location.
    #[wasm_bindgen(method, variadic)]
    pub fn set(this: &TensorBuffer, value: &JsValue, locs: &[usize]);

    /// Creates an immutable Tensor object from the buffer.
    #[wasm_bindgen(method, js_name = "toTensor")]
    pub fn to_tensor(this: &TensorBuffer) -> Tensor;
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A mutable Tensor, useful for persisting state, e.g. for training.
    #[wasm_bindgen(extends = Tensor)]
    pub type Variable;

    #[wasm_bindgen(constructor)]
    pub fn new(that: &Tensor) -> Variable;

    /// Assign a new Tensor to this variable. The new Tensor must have the same shape and dtype as
    /// the old Tensor.
    #[wasm_bindgen(method)]
    pub fn assign(this: &Variable, value: &Tensor);
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Creates rank-0 Tensor (scalar) with the provided value and dtype.
    pub fn scalar(value: &JsValue, dtype: Option<&str>) -> Tensor;

    /// Creates a Tensor with the provided values, shape and dtype.
    pub fn tensor(values: &Array, shape: Option<&Array>, dtype: Option<&str>) -> Tensor;
}
