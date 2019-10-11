use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A Layer is a grouping of operations and weights that can be composed to create a
    /// LayersModel.
    pub type Layer;
}
