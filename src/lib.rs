mod browser;
mod callbacks;
mod constraints;
mod data;
mod initializers;
mod layers;
mod metrics;
mod models;
mod performance;
mod tensors;
mod training;
mod util;

pub use layers::*;
pub use models::*;
pub use performance::*;
pub use tensors::*;
pub use training::*;

use js_sys::JsString;
use wasm_bindgen::prelude::*;

pub type DType = &'static str;

// FIXME
pub type OpMapper = js_sys::Object;

#[wasm_bindgen]
extern {
    pub type Constraint;
}

#[wasm_bindgen]
extern {
    pub type DatasetContainer;
}

#[wasm_bindgen]
extern {
    pub type DisposeResult;

    /// Reference count after the dispose call.
    #[wasm_bindgen(method, getter, js_name = "refCountAfterDispose")]
    pub fn ref_count_after_dispose(this: &DisposeResult) -> usize;

    /// Number of variables dispose in this dispose call.
    #[wasm_bindgen(method, getter, js_name = "numDisposedVariables")]
    pub fn num_disposed_variables(this: &DisposeResult) -> usize;
}

#[wasm_bindgen]
extern {
    pub type Initializer;
}

#[wasm_bindgen]
extern {
    pub type IOHandler;
}

#[wasm_bindgen]
extern {
    pub type Regularizer;
}

#[wasm_bindgen]
extern {
    /// Serializable defines the serialization contract.
    pub type Serializable;

    /// Return the class name for this class to use in serialization contexts.
    #[wasm_bindgen(method)]
    pub fn get_class_name(this: &Serializable) -> JsString;

    /// Return all the non-weight state needed to serialize this object.
    #[wasm_bindgen(method)]
    pub fn get_config(this: &Serializable) -> JsString;

// FIXME
// pub fn from_config(class, config);
}
