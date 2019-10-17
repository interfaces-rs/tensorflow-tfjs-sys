use crate::{Activation, DType};
use js_sys::{Array, Float32Array, JsString, Number, Promise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// MomentsResult is a dictionary returned from `Tensor.moments`.
    pub type MomentsResult;

    #[wasm_bindgen(method, getter)]
    pub fn mean(this: &MomentsResult) -> Tensor;

    #[wasm_bindgen(method, getter)]
    pub fn variance(this: &MomentsResult) -> Tensor;
}

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

    /// Computes absolute value element-wise: abs(x).
    #[wasm_bindgen(method)]
    pub fn abs(this: &Tensor) -> Tensor;

    /// Returns the tensor data as a nested array. The transfer of data is done asynchronously.
    #[wasm_bindgen(method)]
    #[must_use]
    pub fn array(this: &Tensor) -> Promise;

    /// Returns the tensor data as a nested array. The transfer of data is done synchronously.
    #[wasm_bindgen(method, js_name = "arraySync")]
    pub fn array_sync(this: &Tensor) -> Box<[JsValue]>;

    /// Converts a Tensor to a Tensor1D.
    #[wasm_bindgen(method, js_name = "as1D")]
    pub fn as_1d(this: &Tensor) -> Tensor;

    /// Converts a Tensor to a Tensor2D.
    #[wasm_bindgen(method, js_name = "as2D")]
    pub fn as_2d(this: &Tensor, rows: usize, columns: usize) -> Tensor;

    /// Converts a Tensor to a Tensor3D.
    #[wasm_bindgen(method, js_name = "as3D")]
    pub fn as_3d(this: &Tensor, rows: usize, columns: usize, depth: usize) -> Tensor;

    /// Converts a Tensor to a Tensor4D.
    #[wasm_bindgen(method, js_name = "as4D")]
    pub fn as_4d(this: &Tensor, rows: usize, columns: usize, depth: usize, depth2: usize) -> Tensor;

    /// Converts a Tensor to a Tensor5D.
    #[wasm_bindgen(method, js_name = "as5D")]
    pub fn as_5d(this: &Tensor, rows: usize, columns: usize, depth: usize, depth2: usize, depth3: usize) -> Tensor;

    /// Converts a size-1 Tensor to a Scalar.
    #[wasm_bindgen(method, js_name = "asScalar")]
    pub fn as_scalar(this: &Tensor) -> Tensor;

    /// Casts a Tensor to a specified dtype.
    #[wasm_bindgen(method, js_name = "asType")]
    pub fn as_type(this: &Tensor, dtype: DType) -> Tensor;

    /// Returns a promise of TensorBuffer that holds the underlying data.
    #[wasm_bindgen(method)]
    pub fn buffer(this: &Tensor) -> Promise;

    /// Returns a TensorBuffer that holds the underlying data.
    #[wasm_bindgen(method, js_name = "bufferSync")]
    pub fn buffer_sync(this: &Tensor) -> TensorBuffer;

    /// Returns a copy of the tensor. See clone() for details.
    #[wasm_bindgen(method)]
    pub fn clone(this: &Tensor) -> Tensor;

    /// Asynchronously downloads the values from the Tensor. Returns a promise of TypedArray
    /// that resolves when the computation has finished.
    #[wasm_bindgen(method)]
    pub fn data(this: &Tensor) -> Promise;

    /// Synchronously downloads the values from the Tensor. This blocks the UI thread until the
    /// values are ready, which can cause performance issues.
    #[wasm_bindgen(method, js_name = "dataSync")]
    pub fn data_sync(this: &Tensor) -> JsValue;

    /// Disposes Tensor from memory.
    #[wasm_bindgen(method)]
    pub fn dispose(this: &Tensor);

    /// Returns a Tensor that has expanded rank, by inserting a dimension into the tensor's
    /// shape. See expandDims() for details.
    #[wasm_bindgen(method, js_name = "expandDims")]
    pub fn expand_dims(this: &Tensor, axis: Option<usize>) -> Tensor;

    /// Returns the cumulative sum of the Tensor along axis.
    #[wasm_bindgen(method)]
    pub fn cumsum(this: &Tensor, axis: Option<usize>, exclusive: Option<bool>, reverse: Option<bool>) -> Tensor;

    /// Flatten a Tensor to a 1D array.
    #[wasm_bindgen(method)]
    pub fn flatten(this: &Tensor) -> Tensor;

    /// Prints information about the Tensor including its data.
    #[wasm_bindgen(method)]
    pub fn print(this: &Tensor, verbose: Option<bool>);

    /// Reshapes the tensor into the provided shape. See reshape() for more details.
    #[wasm_bindgen(method)]
    pub fn reshape(this: &Tensor, shape: &[i32]) -> Tensor;

    /// Reshapes the tensor into the shape of the provided tensor.
    #[wasm_bindgen(method, js_name = "reshapeAs")]
    pub fn reshape_as(this: &Tensor, that: &Tensor) -> Tensor;

    /// Returns a Tensor with dimensions of size 1 removed from the shape. See squeeze() for
    /// more details.
    #[wasm_bindgen(method)]
    pub fn squeeze(this: &Tensor, axis: Option<&[usize]>) -> Tensor;

    /// Casts the array to type bool
    #[wasm_bindgen(method, js_name = "toBool")]
    pub fn to_bool(this: &Tensor) -> Tensor;

    /// Casts the array to type float32
    #[wasm_bindgen(method, js_name = "toFloat")]
    pub fn to_float(this: &Tensor) -> Tensor;

    /// Casts the array to type int32
    #[wasm_bindgen(method, js_name = "toInt")]
    pub fn to_int(this: &Tensor) -> Tensor;

    /// Returns a human-readable description of the tensor.
    #[wasm_bindgen(method, js_name = "toString")]
    pub fn to_string(this: &Tensor) -> JsString;
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

// *********************
// Arithmetic Operations
// *********************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Adds two Tensors element-wise, A + B.
    pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Adds a list of Tensors element-wise, each with the same shape and dtype.
    #[wasm_bindgen(js_name = "addN")]
    pub fn add_n(tensors: &Array) -> Tensor;

    /// Divides two Tensors element-wise, A / B.
    pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Divides two Tensors element-wise, A / B. Supports broadcasting.
    #[wasm_bindgen(js_name = "floorDiv")]
    pub fn floor_div(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Returns the max of a and b (a > b ? a : b) element-wise.
    pub fn maximum(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Returns the min of a and b (a < b ? a : b) element-wise.
    pub fn minimum(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Multiplies two Tensors element-wise, A * B.
    pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Computes the power of one Tensor to another.
    pub fn pow(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Returns the mod of a and b element-wise. floor(x / y) * y + mod(x, y) = x.
    #[wasm_bindgen(js_name = "mod")]
    pub fn modulus(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Returns (a - b) * (a - b) element-wise.
    #[wasm_bindgen(js_name = "squaredDifference")]
    pub fn squared_difference(lhs: &Tensor, rhs: &Tensor) -> Tensor;

    /// Subtracts two Tensors element-wise, A - B.
    pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor;
}

// *********************
// Basic Math Operations
// *********************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Computes absolute value element-wise: abs(x)
    pub fn abs(x: &Tensor) -> Tensor;

    /// Computes acos of the input Tensor element-wise: acos(x)
    pub fn acos(x: &Tensor) -> Tensor;

    /// Computes the inverse hyperbolic cos of the input Tensor element-wise: acosh(x)
    #[wasm_bindgen(js_name = "acos_h")]
    pub fn acos_h(x: &Tensor) -> Tensor;

    /// Computes asin of the input Tensor element-wise: asin(x)
    pub fn asin(x: &Tensor) -> Tensor;

    /// Computes inverse hyperbolic sin of the input Tensor element-wise: asinh(x)
    pub fn asin_h(x: &Tensor) -> Tensor;

    /// Computes atan of the input Tensor element-wise: atan(x)
    pub fn atan(x: &Tensor) -> Tensor;

    /// Computes arctangent of Tensors a / b element-wise: atan2(a, b).
    pub fn atan2(x: &Tensor) -> Tensor;

    /// Computes inverse hyperbolic tan of the input Tensor element-wise: atanh(x)
    #[wasm_bindgen(js_name = "atanh")]
    pub fn atan_h(x: &Tensor) -> Tensor;

    /// Computes ceiling of input Tensor element-wise: ceil(x)
    pub fn ceil(x: &Tensor) -> Tensor;

    /// Clips values element-wise. max(min(x, clipValueMax), clipValueMin)
    pub fn clip_by_value(x: &Tensor, clip_min: i32, clip_max: i32) -> Tensor;

    /// Computes cos of the input Tensor element-wise: cos(x)
    pub fn cos(x: &Tensor) -> Tensor;

    /// Computes hyperbolic cos of the input Tensor element-wise: cosh(x)
    #[wasm_bindgen(js_name = "cosh")]
    pub fn cos_h(x: &Tensor) -> Tensor;

    /// Computes exponential linear element-wise: x > 0 ? e ^ x - 1 : 0.
    pub fn elu(x: &Tensor) -> Tensor;

    /// Computes gause error function of the input Tensor element-wise: erf(x)
    pub fn erf(x: &Tensor) -> Tensor;

    /// Computes exponential of the input Tensor element-wise. e ^ x
    pub fn exp(x: &Tensor) -> Tensor;

    /// Computes exponential of the input Tensor minus one element-wise. e ^ x - 1
    #[wasm_bindgen(js_name = "expm1")]
    pub fn exp_m1(x: &Tensor) -> Tensor;

    /// Computes floor of input Tensor element-wise: floor(x).
    pub fn floor(x: &Tensor) -> Tensor;

    /// Returns which elements of x are finite.
    #[wasm_bindgen(js_name = "isFinite")]
    pub fn is_finite(x: &Tensor) -> Tensor;

    /// Returns which elements of x are Infinity or -Infinity.
    #[wasm_bindgen(js_name = "isInf")]
    pub fn is_inf(x: &Tensor) -> Tensor;

    /// Returns which elements of x are NaN.
    #[wasm_bindgen(js_name = "isNaN")]
    pub fn is_nan(x: &Tensor) -> Tensor;

    /// Computes leaky rectified linear element-wise.
    #[wasm_bindgen(js_name = "leakyRelu")]
    pub fn leaky_relu(x: &Tensor, alpha: i32) -> Tensor;

    /// Computes natural logarithm of the input Tensor element-wise: ln(x)
    pub fn log(x: &Tensor) -> Tensor;

    /// Computes natural logarithm of the input Tensor plus one element-wise: ln(1 + x)
    #[wasm_bindgen(js_name = "log1p")]
    pub fn log_1p(x: &Tensor) -> Tensor;

    /// Computes log sigmoid of the input Tensor element-wise: logSigmoid(x). For numerical
    /// stability, we use -softplus(-x).
    #[wasm_bindgen(js_name = "logSigmoid")]
    pub fn log_sigmoid(x: &Tensor) -> Tensor;

    /// Computes -1 * x element-wise.
    pub fn neg(x: &Tensor) -> Tensor;

    /// Computes leaky rectified linear element-wise with parametric alphas.
    pub fn prelu(x: &Tensor, alpha: &Tensor) -> Tensor;

    /// Computes reciprocal of x element-wise: 1 / x
    pub fn reciprocal(x: &Tensor) -> Tensor;

    /// Computes rectified linear element-wise: max(x, 0).
    pub fn relu(x: &Tensor) -> Tensor;

    /// Computes round of input Tensor element-wise: round(x).
    pub fn round(x: &Tensor) -> Tensor;

    /// Computes reciprocal of square root of the input Tensor element-wise: y = 1 / sqrt(x)
    pub fn rsqrt(x: &Tensor) -> Tensor;

    /// Computes scaled exponential linear element-wise.
    pub fn selu(x: &Tensor) -> Tensor;

    /// Computes sigmoid element-wise, 1 / (1 + exp(-x))
    pub fn sigmoid(x: &Tensor) -> Tensor;

    /// Returns an element-wise indication of the sign of a number.
    pub fn sign(x: &Tensor) -> Tensor;

    /// Computes sin of the input Tensor element-wise: sin(x)
    pub fn sin(x: &Tensor) -> Tensor;

    /// Computes hyperbolic sin of the input Tensor element-wise: sinh(x)
    #[wasm_bindgen(js_name = "sinh")]
    pub fn sin_h(x: &Tensor) -> Tensor;

    /// Computes softplus of the input Tensor element-wise: log(exp(x) + 1)
    #[wasm_bindgen(js_name = "softplus")]
    pub fn soft_plus(x: &Tensor) -> Tensor;

    /// Computes square root of the input Tensor element-wise: y = sqrt(x)
    pub fn sqrt(x: &Tensor) -> Tensor;

    /// Computes square of x element-wise: x ^ 2
    pub fn square(x: &Tensor) -> Tensor;

    /// Computes step of the input Tensor element-wise: x > 0 ? 1 : alpha * x
    pub fn step(x: &Tensor, alpha: Option<i32>) -> Tensor;

    /// Computes tan of the input Tensor element-wise, tan(x)
    pub fn tan(x: &Tensor) -> Tensor;

    /// Computes hyperbolic tangent of the input Tensor element-wise: tanh(x)
    #[wasm_bindgen(js_name = "tanH")]
    pub fn tan_h(x: &Tensor) -> Tensor;
}

// *****************
// Matrix Operations
// *****************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Computes the dot product of two matrices and/or vectors, t1 and t2.
    pub fn dot(t1: &Tensor, t2: &Tensor) -> Tensor;

    /// Computes the dot product of two matrices, A * B. These must be matrices.
    pub fn mat_mul(
        a: &Tensor,
        b: &Tensor,
        transposeA: Option<bool>,
        transposeB: Option<bool>,
        bias: &Tensor,
        activation: &Activation,
    ) -> Tensor;

    /// Computes the norm of scalar, vectors, and matrices.
    pub fn norm(x: &Tensor, ord: &JsValue, axis: Option<&[i32]>, keep_dims: Option<bool>) -> Tensor;

    /// Computes the outer product of two vectors, v1 and v2.
    pub fn outer_product(v1: &Tensor, v2: &Tensor) -> Tensor;

    /// Transposes the Tensor. Permutes the dimensions according to perm.
    pub fn transpose(x: &Tensor, perm: Option<&[usize]>) -> Tensor;
}

// **********************
// Convolution Operations
// **********************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Computes the 2D average pooling of an image.
    #[wasm_bindgen(js_name = "avgPool")]
    pub fn avg_pool(
        x: &Tensor,
        filter_size: &[usize],
        strides: &[usize],
        pad: &JsValue,
        dim_rounding_mode: &str,
    ) -> Tensor;

    /// Computes a 1D convolution over the input x.
    #[wasm_bindgen(js_name = "conv1d")]
    pub fn conv_1d(
        x: &Tensor,
        filter: &Tensor,
        stride: usize,
        pad: &JsValue,
        data_format: Option<&str>,
        dilation: Option<usize>,
        dim_rounding_mode: Option<&str>,
    ) -> Tensor;

    /// Computes a 2D convolution over the input x.
    #[wasm_bindgen(js_name = "conv2d")]
    pub fn conv_2d(
        x: &Tensor,
        filter: &Tensor,
        stride: usize,
        pad: &JsValue,
        data_format: Option<&str>,
        dilation: Option<usize>,
        dim_rounding_mode: Option<&str>,
    ) -> Tensor;

    /// Computes the transposed 2D convolution of an image, also known as a deconvolution.
    #[wasm_bindgen(js_name = "conv2dTranspose")]
    pub fn conv_2d_transpose(
        x: &Tensor,
        filter: &Tensor,
        stride: usize,
        pad: &JsValue,
        data_format: Option<&str>,
        dilation: Option<usize>,
        dim_rounding_mode: Option<&str>,
        bias: Option<&Tensor>,
        activation: &Activation,
    ) -> Tensor;

    /// Computes a 3D convolution over the input x.
    #[wasm_bindgen(js_name = "conv3d")]
    pub fn conv_3d(
        x: &Tensor,
        filter: &Tensor,
        stride: usize,
        pad: &JsValue,
        data_format: Option<&str>,
        dilation: Option<usize>,
    );

    /// Depthwise 2D convolution.
    #[wasm_bindgen(js_name = "depthwiseConv2d")]
    pub fn depthwise_conv_2d(
        x: &Tensor,
        filter: &Tensor,
        stride: usize,
        pad: &JsValue,
        data_format: Option<&str>,
        dilation: Option<usize>,
        dim_rounding_mode: Option<&str>,
    ) -> Tensor;

    /// Computes the 2D max pooling of an image.
    #[wasm_bindgen(js_name = "maxPool")]
    pub fn max_pool(
        x: &Tensor,
        filter_size: &[usize],
        strides: &[usize],
        pad: &JsValue,
        dim_rounding_mode: &str,
    ) -> Tensor;

    /// Performs an N-D pooling operation.
    pub fn pool(
        input: &Tensor,
        window_shape: &[usize],
        pooling_type: &str,
        pad: &JsValue,
        dilations: Option<&[usize]>,
        strides: Option<&[usize]>,
    ) -> Tensor;

    /// 2-D convolution with separable filters.
    #[wasm_bindgen(js_name = "separableConv2d")]
    pub fn separable_conv_2d(
        x: &Tensor,
        depthwise: &Tensor,
        pointwise: &Tensor,
        strides: &[usize],
        pad: &JsValue,
        dilation: Option<&[usize]>,
        data_format: Option<&str>,
    ) -> Tensor;
}

// ********************
// Reduction Operations
// ********************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Computes the logical and of elements across dimensions of a Tensor.
    pub fn all(x: &Tensor, axis: Option<&[usize]>, keep_dims: Option<bool>) -> Tensor;

    /// Computes the logical or of elements across dimensions of a Tensor.
    pub fn any(x: &Tensor, axis: Option<&[usize]>, keep_dims: Option<bool>) -> Tensor;

    /// Returns the indices of the maximum values along an axis.
    #[wasm_bindgen(js_name = "argMax")]
    pub fn arg_max(x: &Tensor, axis: Option<&[usize]>) -> Tensor;

    /// Returns the indices of the minimum values along an axis.
    #[wasm_bindgen(js_name = "argMin")]
    pub fn arg_min(x: &Tensor, axis: Option<&[usize]>) -> Tensor;

    /// Computes the log(sum(exp(elements across the reduction dimensions)).
    #[wasm_bindgen(js_name = "logSumExp")]
    pub fn log_sum_exp(x: &Tensor, axis: Option<&[usize]>, keep_dims: Option<bool>) -> Tensor;

    /// Computes the maximum of elements across dimensions of a Tensor.
    pub fn max(x: &Tensor, axis: Option<&[usize]>, keep_dims: Option<bool>) -> Tensor;

    /// Computes the mean of elements across dimensions of a Tensor.
    pub fn mean(x: &Tensor, axis: Option<&[usize]>, keep_dims: Option<bool>) -> Tensor;

    /// Computes the minimum value from the input.
    pub fn min(x: &Tensor, axis: Option<&[usize]>, keep_dims: Option<bool>) -> Tensor;

    /// Computes the product of elements across dimensions of a Tensor.
    pub fn prod(x: &Tensor, axis: Option<&[usize]>, keep_dims: Option<bool>) -> Tensor;

    /// Computes the sum of elements across dimensions of a Tensor.
    pub fn sum(x: &Tensor, axis: Option<&[usize]>, keep_dims: Option<bool>) -> Tensor;
}

// ************************
// Normalization Operations
// ************************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Batch normalization.
    #[wasm_bindgen(js_name = "batchNorm")]
    pub fn batch_norm(
        x: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        offset: Option<&Tensor>,
        scale: Option<&Tensor>,
        variance_epsilon: Option<f32>,
    ) -> Tensor;

    /// Normalizes the activation of a local neighborhood across or within channels.
    #[wasm_bindgen(js_name = "localResponseNormalization")]
    pub fn local_response_normalization(
        x: &Tensor,
        depth_radius: Option<usize>,
        bias: Option<i32>,
        alpha: Option<i32>,
        beta: Option<i32>,
    ) -> Tensor;

    /// Computes the log softmax.
    #[wasm_bindgen(js_name = "logSoftmax")]
    pub fn log_softmax(logits: &Tensor, axis: Option<i32>) -> Tensor;

    /// Calculates the mean and variance of x.
    pub fn moments(x: &Tensor, axis: Option<&[i32]>, keep_dims: Option<bool>) -> MomentsResult;

    /// Computes the softmax normalized vector given the logits.
    pub fn softmax(logits: &Tensor, dim: Option<i32>) -> Tensor;

    /// Converts a sparse representation into a dense tensor.
    #[wasm_bindgen(js_name = "sparseToDense")]
    pub fn sparse_to_dense(
        sparse_indices: &Tensor,
        sparse_values: &Tensor,
        output_shape: &[i32],
        default_value: Option<&Tensor>,
    ) -> Tensor;
}

// ******************
// Logical Operations
// ******************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Returns the truth value of (a == b) element-wise.
    pub fn equal(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the truth value of (a > b) element-wise.
    pub fn greater(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the truth value of (a >= b) element-wise.
    #[wasm_bindgen(js_name = "greaterEqual")]
    pub fn greater_equal(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the truth value of (a < b) element-wise.
    pub fn less(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the truth value of (a <= b) element-wise.
    #[wasm_bindgen(js_name = "lessEqual")]
    pub fn less_equal(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the truth value of a AND b element-wise.
    #[wasm_bindgen(js_name = "logicalAnd")]
    pub fn logical_and(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the truth value of a OR b element-wise.
    #[wasm_bindgen(js_name = "logicalOr")]
    pub fn logical_or(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the truth value of a XOR b element-wise.
    #[wasm_bindgen(js_name = "logicalXor")]
    pub fn logical_xor(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the truth value of (a != b) element-wise.
    #[wasm_bindgen(js_name = "notEqual")]
    pub fn not_equal(a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the elements, either a or b depending on the condition.
    pub fn where_(condition: &Tensor, a: &Tensor, b: &Tensor) -> Tensor;

    /// Returns the coordinates of true elements of condition.
    #[wasm_bindgen(js_name = "whereAsync")]
    pub fn where_async(condition: &Tensor) -> Promise;
}

// ******************
// Creation Functions
// ******************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Creates an empty TensorBuffer with the specified shape and dtype.
    pub fn buffer(shape: &[i32], dtype: Option<DType>, values: Option<&Float32Array>) -> TensorBuffer;

    /// Creates a new tensor with the same values and shape as the specified tensor.
    pub fn clone(tensor: &Tensor) -> Tensor;

    /// Converts two real numbers to a complex number.
    pub fn complex(real: f32, imag: f32) -> Tensor;

    /// Create an identity matrix.
    pub fn eye(
        num_rows: usize,
        num_columns: Option<usize>,
        batch_shape: Option<&[usize]>,
        dtype: Option<DType>,
    ) -> Tensor;

    /// Creates a Tensor filled with a scalar value.
    pub fn fill(shape: &[i32], value: i32, dtype: Option<DType>) -> Tensor;

    /// Returns the imaginary part of a complex (or real) tensor.
    pub fn imag(input: &Tensor) -> Tensor;

    /// Return an evenly spaced sequence of numbers over the given interval.
    pub fn linspace(start: i32, stop: i32, num: usize) -> Tensor;

    /// Creates a one-hot Tensor.
    #[wasm_bindgen(js_name = "oneHot")]
    pub fn one_hot(indices: &Tensor, depth: usize, on_value: i32, off_value: i32) -> Tensor;

    /// Creates a Tensor with all elements set to 1.
    pub fn ones(shape: &[i32], dtype: Option<DType>) -> Tensor;

    /// Creates a Tensor with all elements set to 1 with the same shape as the given tensor.
    #[wasm_bindgen(js_name = "onesLike")]
    pub fn ones_like(x: &Tensor) -> Tensor;

    /// Prints information about the Tensor including its data.
    pub fn print(x: &Tensor, verbose: Option<bool>);

    /// Creates a new Tensor1D filled with the numbers in the range provided.
    pub fn range(start: i32, stop: i32, step: Option<i32>, dtype: Option<DType>) -> Tensor;

    /// Returns the real part of a complex (or real) tensor.
    pub fn real(input: &Tensor) -> Tensor;

    /// Creates rank-0 Tensor (scalar) with the provided value and dtype.
    pub fn scalar(value: f32, dtype: Option<DType>) -> Tensor;

    /// Creates a Tensor with the provided values, shape and dtype.
    pub fn tensor(values: &Array, shape: Option<&[i32]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-1 Tensor with the provided values, shape and dtype.
    pub fn tensor1d(values: &Array, dtype: Option<DType>) -> Tensor;

    /// Creates rank-2 Tensor with the provided values, shape and dtype.
    pub fn tensor2d(values: &Array, shape: Option<&[i32]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-3 Tensor with the provided values, shape and dtype.
    pub fn tensor3d(values: &Array, shape: Option<&[i32]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-4 Tensor with the provided values, shape and dtype.
    pub fn tensor4d(values: &Array, shape: Option<&[i32]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-5 Tensor with the provided values, shape and dtype.
    pub fn tensor5d(values: &Array, shape: Option<&[i32]>, dtype: Option<DType>) -> Tensor;

    /// Creates rank-6 Tensor with the provided values, shape and dtype.
    pub fn tensor6d(values: &Array, shape: Option<&[i32]>, dtype: Option<DType>) -> Tensor;

    /// Creates a Tensor with values sampled from a truncated normal distribution.
    #[wasm_bindgen(js_name = "truncatedNormal")]
    pub fn truncated_normal(
        shape: &[i32],
        mean: Option<f32>,
        std_dev: Option<f32>,
        dtype: Option<DType>,
        seed: Option<f32>,
    ) -> Tensor;

    /// Creates a new variable with the provided initial value.
    pub fn variable(
        initial_value: &Tensor,
        trainable: Option<bool>,
        name: Option<&JsString>,
        dtype: Option<DType>,
    ) -> Variable;

    /// Creates a Tensor with all elements set to 0.
    pub fn zeros(shape: &[i32], dtype: Option<DType>) -> Tensor;

    /// Creates a Tensor with all elements set to 0 with the same shape as the given tensor.
    #[wasm_bindgen(js_name = "zerosLike")]
    pub fn zeros_like(input: &Tensor) -> Tensor;
}

// ************************
// Transformation Functions
// ************************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// This operation reshapes the "batch" dimension 0 into M + 1 dimensions of shape blockShape +
    /// [batch], interleaves these blocks back into the grid defined by the spatial dimensions [1,
    /// ..., M], to obtain a result with the same rank as the input.
    // FIXME: crops type
    #[wasm_bindgen(js_name = "batchToSpaceND")]
    pub fn batch_to_space_nd(x: &Tensor, block_shape: &[usize], crops: &Array) -> Tensor;

    /// Casts a Tensor to a new dtype.
    pub fn cast(x: &Tensor, dtype: DType) -> Tensor;

    /// Rearranges data from depth into blocks of spatial data.
    #[wasm_bindgen(js_name = "depthToSpace")]
    pub fn depth_to_space(x: &Tensor, block_size: usize, data_format: Option<&str>) -> Tensor;

    /// Returns a Tensor that has expanded rank, by inserting a dimension into the tensor's shape.
    #[wasm_bindgen(js_name = "depthToSpace")]
    pub fn expand_dims(x: &Tensor, axis: Option<usize>) -> Tensor;

    /// Pads a Tensor with a given value and paddings.
    pub fn pad(x: &Tensor, paddings: &Array, constant_value: f32) -> Tensor;

    /// Reshapes a Tensor to a given shape.
    pub fn reshape(x: &Tensor, shape: &[i32]) -> Tensor;

    /// Computes the difference between two lists of numbers.
    #[wasm_bindgen(js_name = "setdiff1dAsync")]
    #[must_use]
    pub fn set_diff_1d_async(x: &Tensor, y: &Tensor) -> Promise;

    /// This operation divides "spatial" dimensions [1, ..., M] of the input into a grid of blocks
    /// of shape blockShape, and interleaves these blocks with the "batch" dimension (0) such that
    /// in the output, the spatial dimensions [1, ..., M] correspond to the position within the
    /// grid, and the batch dimension combines both the position within a spatial block and the
    /// original batch position. Prior to division into blocks, the spatial dimensions of the input
    /// are optionally zero padded according to paddings.
    #[wasm_bindgen(js_name = "spaceToBatchND")]
    pub fn space_to_batch_nd(x: &Tensor, shape: &[i32], paddings: &Array) -> Tensor;

    /// Removes dimensions of size 1 from the shape of a Tensor.
    pub fn squeeze(x: &Tensor, axis: &[usize]) -> Tensor;
}

// Slicing and Joining
#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Concatenates a list of Tensors along a given axis.
    pub fn concat(tensors: &Array, axis: Option<usize>) -> Tensor;

    /// Gather slices from tensor x's axis axis according to indices.
    pub fn gather(x: &Tensor, indices: &Tensor, axis: Option<usize>) -> Tensor;

    /// Reverses a Tensor along a specified axis.
    pub fn reverse(x: &Tensor, axis: Option<usize>) -> Tensor;

    /// Extracts a slice from a Tensor starting at coordinates begin and is of size size.
    pub fn slice(x: &Tensor, begin: &JsValue, end: &JsValue) -> Tensor;

    /// Splits a Tensor into sub tensors.
    pub fn split(x: &Tensor, num_or_size_splits: &JsValue, axis: Option<usize>) -> Array;

    /// Stacks a list of rank-R Tensors into one rank-(R+1) Tensor.
    pub fn stack(tensors: &Array, axis: usize) -> Tensor;

    /// Construct a tensor by repeating it the number of times given by reps.
    pub fn tile(x: &Tensor, reps: &[usize]) -> Tensor;

    /// Unstacks a Tensor of rank-R into a list of rank-(R-1) Tensors.
    pub fn unstack(x: &Tensor, axis: Option<usize>) -> Array;
}

// ****************
// Random Functions
// ****************

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Creates a Tensor with values drawn from a multinomial distribution.
    // FIXME: seed type?
    pub fn multinomial(logits: &Tensor, num_samples: usize, seed: Option<usize>, normalized: Option<bool>) -> Tensor;

    /// Creates a Tensor with values sampled from a normal distribution.
    // FIXME: seed type?
    #[wasm_bindgen(js_name = "randomNormal")]
    pub fn random_normal(
        shape: &[i32],
        mean: Option<&Number>,
        std_dev: Option<&Number>,
        dtype: Option<DType>,
        seed: Option<usize>,
    ) -> Tensor;

    /// Creates a Tensor with values sampled from a uniform distribution.
    #[wasm_bindgen(js_name = "randomUniform")]
    pub fn random_uniform(
        shape: &[i32],
        minval: Option<&Number>,
        maxval: Option<&Number>,
        dtype: Option<DType>,
        seed: Option<usize>,
    ) -> Tensor;
}
