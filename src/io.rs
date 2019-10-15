use js_sys::Promise;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Copy a model from one URL to another.
    #[must_use]
    pub fn copy_model(src: &str, dst: &str) -> Promise;

    /// List all models stored in registered storage mediums.
    #[must_use]
    pub fn list_models() -> Promise;

    /// Move a model from one URL to another.
    #[must_use]
    pub fn move_model(src: &str, dst: &str) -> Promise;

    /// Remove a model specified by URL from a reigstered storage medium.
    #[must_use]
    pub fn remove_model(url: &str) -> Promise;
}
