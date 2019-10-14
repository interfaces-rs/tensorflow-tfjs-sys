use js_sys::{Function, Object};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Factory function for a Callback that stops training when a monitored quantity has stopped
    /// improving.
    #[wasm_bindgen(js_name = "earlyStopping")]
    pub fn early_stopping(args: Option<&Object>) -> Function;
}
