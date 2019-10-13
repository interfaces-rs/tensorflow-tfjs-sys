
use crate::Serializable;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern {
    /// Base class for functions that impose constraints on weight values.
    #[wasm_bindgen(extends = Serializable)]
    pub type Constraint;
}
