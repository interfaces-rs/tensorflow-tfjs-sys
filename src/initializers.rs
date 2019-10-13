use crate::Serializable;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern {
    /// Initializer base class.
    #[wasm_bindgen(extends = Serializable)]
    pub type Initializer;
}
