use crate::DType;
use js_sys::{Array, Float32Array, JsString, Number};
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
    /// Creates an empty TensorBuffer with the specified shape and dtype.
    pub fn buffer(shape: &Array, dtype: Option<DType>, values: Option<&Float32Array>) -> TensorBuffer;

    /// Creates a new tensor with the same values and shape as the specified tensor.
    pub fn clone(x: &JsValue) -> Tensor;

    /// Converts two real numbers to a complex number.
    pub fn complex(real: &JsValue, imag: &JsValue) -> Tensor;

    /// Create an identity matrix.
    pub fn eye(
        num_rows: usize,
        num_columns: Option<usize>,
        batch_shape: Option<&[usize]>,
        dtype: Option<DType>,
    ) -> Tensor;

    /// Creates a Tensor filled with a scalar value.
    pub fn fill(shape: &[usize], value: &JsValue, dtype: Option<DType>) -> Tensor;

    /// Returns the imaginary part of a complex (or real) tensor.
    pub fn imag(input: &JsValue) -> Tensor;

    /// Return an evenly spaced sequence of numbers over the given interval.
    pub fn linspace(start: &Number, stop: &Number, num: usize) -> Tensor;

    /// Creates a one-hot Tensor.
    #[wasm_bindgen(js_name = "oneHot")]
    pub fn one_hot(indices: &JsValue, depth: usize, on_value: &Number, off_value: &Number) -> Tensor;

    /// Creates a Tensor with all elements set to 1.
    pub fn ones(shape: &[usize], dtype: Option<DType>) -> Tensor;

    /// Creates a Tensor with all elements set to 1 with the same shape as the given tensor.
    #[wasm_bindgen(js_name = "onesLike")]
    pub fn ones_like(x: &JsValue) -> Tensor;

    /// Prints information about the Tensor including its data.
    pub fn print(x: &Tensor, verbose: Option<bool>);

    /// Creates a new Tensor1D filled with the numbers in the range provided.
    pub fn range(start: i32, stop: i32, step: i32, dtype: Option<DType>) -> Tensor;

    /// Returns the real part of a complex (or real) tensor.
    pub fn real(input: &JsValue) -> Tensor;

    /// Creates rank-0 Tensor (scalar) with the provided value and dtype.
    pub fn scalar(value: &JsValue, dtype: Option<DType>) -> Tensor;

    /// Creates a Tensor with the provided values, shape and dtype.
    pub fn tensor(values: &Array, shape: Option<&[usize]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-1 Tensor with the provided values, shape and dtype.
    pub fn tensor1d(values: &Array, dtype: Option<DType>) -> Tensor;

    /// Creates rank-2 Tensor with the provided values, shape and dtype.
    pub fn tensor2d(values: &Array, shape: Option<&[usize]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-3 Tensor with the provided values, shape and dtype.
    pub fn tensor3d(values: &Array, shape: Option<&[usize]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-4 Tensor with the provided values, shape and dtype.
    pub fn tensor4d(values: &Array, shape: Option<&[usize]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-5 Tensor with the provided values, shape and dtype.
    pub fn tensor5d(values: &Array, shape: Option<&[usize]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-6 Tensor with the provided values, shape and dtype.
    pub fn tensor6d(values: &Array, shape: Option<&[usize]>, dtype: Option<DType>) -> Tensor;

    /// Creates a Tensor with values sampled from a truncated normal distribution.
    #[wasm_bindgen(js_name = "truncatedNormal")]
    pub fn truncated_normal(
        shape: &[i32],
        mean: Option<&Number>,
        std_dev: Option<&Number>,
        dtype: Option<DType>,
        seed: Option<&Number>,
    ) -> Tensor;

    /// Creates a new variable with the provided initial value.
    pub fn variable(
        initial_value: &Tensor,
        trainable: Option<bool>,
        name: Option<&JsString>,
        dtype: Option<DType>,
    ) -> Variable;

    /// Creates a Tensor with all elements set to 0.
    pub fn zeroes(shape: &[u32], dtype: Option<DType>) -> Tensor;

    /// Creates a Tensor with all elements set to 0 with the same shape as the given tensor.
    pub fn zeroes_like(input: &JsValue) -> Tensor;
}
