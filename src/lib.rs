mod constraints;
mod data;
mod initializers;
mod layers;
mod models;
mod tensors;
mod training;

pub use constraints::*;
pub use data::*;
pub use initializers::*;
pub use layers::*;
pub use models::*;
pub use tensors::*;
pub use training::*;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern {
    pub type Constraint;
}

#[wasm_bindgen]
extern {
    pub type DisposeResult;

    /// Reference count after the dispose call.
    #[wasm_bindgen(method, getter, js_name = "refCountAfterDispose")]
    fn ref_count_after_dispose(this: &DisposeResult) -> usize;

    /// Number of variables dispose in this dispose call.
    #[wasm_bindgen(method, getter, js_name = "numDisposedVariables")]
    fn num_disposed_variables(this: &DisposeResult) -> usize;
}
