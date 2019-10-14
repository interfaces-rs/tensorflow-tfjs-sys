use crate::DType;
use js_sys::{Array, Float32Array, JsString, Number, Promise};
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

    /// Converts a Tensor to a Tensor1D.
    #[wasm_bindgen(method)]
    pub fn as_1D(this: &Tensor) -> Tensor;

    /// Converts a Tensor to a Tensor2D.
    #[wasm_bindgen(method)]
    pub fn as_2D(this: &Tensor) -> Tensor;

    /// Converts a Tensor to a Tensor3D.
    #[wasm_bindgen(method)]
    pub fn as_3D(this: &Tensor) -> Tensor;

    /// Converts a Tensor to a Tensor4D.
    #[wasm_bindgen(method)]
    pub fn as_4D(this: &Tensor) -> Tensor;

    /// Converts a Tensor to a Tensor5D.
    #[wasm_bindgen(method)]
    pub fn as_5D(this: &Tensor) -> Tensor;

    /// Converts a size-1 Tensor to a Scalar.
    #[wasm_bindgen(method)]
    pub fn as_scalar(this: &Tensor) -> Tensor;

    /// Casts a Tensor to a specified dtype.
    #[wasm_bindgen(method)]
    pub fn as_type(this: &Tensor, dtype: DType) -> Tensor;

    /// Returns the tensor data as a nested array. The transfer of data is done asynchronously.
    #[wasm_bindgen(method)]
    pub fn array(this: &Tensor) -> Promise;

    /// Returns the tensor data as a nested array. The transfer of data is done synchronously.
    #[wasm_bindgen(method)]
    pub fn array_sync(this: &Tensor) -> Box<[JsValue]>;

    /// Returns a promise of TensorBuffer that holds the underlying data.
    #[wasm_bindgen(method)]
    pub fn buffer(this: &Tensor);

    /// Returns a TensorBuffer that holds the underlying data.
    #[wasm_bindgen(method)]
    pub fn buffer_sync(this: &Tensor);

    /// Returns a copy of the tensor. See clone() for details.
    #[wasm_bindgen(method)]
    pub fn clone(this: &Tensor);

    /// Asynchronously downloads the values from the Tensor. Returns a promise of TypedArray
    /// that resolves when the computation has finished.
    #[wasm_bindgen(method)]
    pub fn data(this: &Tensor);

    /// Synchronously downloads the values from the Tensor. This blocks the UI thread until the
    /// values are ready, which can cause performance issues.
    #[wasm_bindgen(method)]
    pub fn data_sync(this: &Tensor);

    /// Disposes Tensor from memory.
    #[wasm_bindgen(method)]
    pub fn dispose(this: &Tensor);

    /// Returns a Tensor that has expanded rank, by inserting a dimension into the tensor's
    /// shape. See expandDims() for details.
    #[wasm_bindgen(method)]
    pub fn expand_dims(this: &Tensor);

    /// Returns the cumulative sum of the Tensor along axis.
    #[wasm_bindgen(method)]
    pub fn cumsum(this: &Tensor);

    /// Flatten a Tensor to a 1D array.
    #[wasm_bindgen(method)]
    pub fn flatten(this: &Tensor);

    /// Prints information about the Tensor including its data.
    #[wasm_bindgen(method)]
    pub fn print(this: &Tensor, verbose: bool);

    /// Reshapes the tensor into the provided shape. See reshape() for more details.
    #[wasm_bindgen(method)]
    pub fn reshape(this: &Tensor);

    /// Reshapes the tensor into the shape of the provided tensor.
    #[wasm_bindgen(method)]
    pub fn reshape_as(this: &Tensor);

    /// Returns a Tensor with dimensions of size 1 removed from the shape. See squeeze() for
    /// more details.
    #[wasm_bindgen(method)]
    pub fn squeeze(this: &Tensor);

    /// Casts the array to type bool
    #[wasm_bindgen(method)]
    pub fn to_bool(this: &Tensor);

    /// Casts the array to type float32
    #[wasm_bindgen(method)]
    pub fn to_float(this: &Tensor);

    /// Casts the array to type int32
    #[wasm_bindgen(method)]
    pub fn to_int(this: &Tensor);
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A mutable object, similar to Tensor, that allows users to set values at locations before
    /// converting to an immutable Tensor.
    pub type TensorBuffer;

    /// Returns the value in the buffer at the provided location.
    #[wasm_bindgen(method, variadic)]
    pub fn get(this: &TensorBuffer, locs: &[usize]) -> JsValue;

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

    /// Assign a new Tensor to this variable. The new Tensor must have the same shape and dtype
    /// as the old Tensor.
    #[wasm_bindgen(method)]
    pub fn assign(this: &Variable, value: &Tensor);
}

// Creation functions
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

// Transformations
#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// This operation reshapes the "batch" dimension 0 into M + 1 dimensions of shape blockShape +
    /// [batch], interleaves these blocks back into the grid defined by the spatial dimensions [1,
    /// ..., M], to obtain a result with the same rank as the input.
    // FIXME: crops type
    #[wasm_bindgen(js_name = "batchToSpaceND")]
    pub fn batch_to_space_nd(x: &JsValue, block_shape: &[usize], crops: &Array) -> Tensor;

    /// Casts a Tensor to a new dtype.
    pub fn cast(x: &JsValue, dtype: DType) -> Tensor;

    /// Rearranges data from depth into blocks of spatial data.
    #[wasm_bindgen(js_name = "depthToSpace")]
    pub fn depth_to_space(x: &JsValue, block_size: usize, data_format: Option<&str>) -> Tensor;

    /// Returns a Tensor that has expanded rank, by inserting a dimension into the tensor's shape.
    #[wasm_bindgen(js_name = "depthToSpace")]
    pub fn expand_dims(x: &JsValue, axis: Option<usize>) -> Tensor;

    /// Pads a Tensor with a given value and paddings.
    pub fn pad(x: &JsValue, paddings: &Array, constant_value: &Number) -> Tensor;

    /// Reshapes a Tensor to a given shape.
    pub fn reshape(x: &JsValue, shape: &[i32]) -> Tensor;

    /// Computes the difference between two lists of numbers.
    #[wasm_bindgen(js_name = "setdiff1dAsync")]
    pub fn set_diff_1d_async(x: &JsValue, y: &JsValue) -> Promise;

    /// This operation divides "spatial" dimensions [1, ..., M] of the input into a grid of blocks
    /// of shape blockShape, and interleaves these blocks with the "batch" dimension (0) such that
    /// in the output, the spatial dimensions [1, ..., M] correspond to the position within the
    /// grid, and the batch dimension combines both the position within a spatial block and the
    /// original batch position. Prior to division into blocks, the spatial dimensions of the input
    /// are optionally zero padded according to paddings.
    #[wasm_bindgen(js_name = "spaceToBatchND")]
    pub fn space_to_batch_nd(x: &JsValue, block_shape: &[usize], paddings: &Array) -> Tensor;

    /// Removes dimensions of size 1 from the shape of a Tensor.
    pub fn squeeze(x: &JsValue, axis: &[usize]) -> Tensor;
}
