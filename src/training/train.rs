use crate::Serializable;
use crate::Tensor;
use js_sys::Function;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern {
    #[wasm_bindgen(extends = Serializable)]
    pub type Optimizer;

    /// Executes f() and minimizes the scalar output of f() by computing gradients of y with respect
    /// to the list of trainable variables provided by varList. If no list is provided, it defaults
    /// to all trainable variables.
    pub fn minimize(f: &Function, return_cost: bool, var_list: Box<[JsValue]>) -> Option<Tensor>;
}
