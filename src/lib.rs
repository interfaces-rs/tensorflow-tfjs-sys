use js_sys::{Array, Promise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A Tensor object represents an immutable, multidimensional array of numbers that has a
    /// shape and a data type.
    #[wasm_bindgen(extends = Promise)]
    pub type Tensor;

}
