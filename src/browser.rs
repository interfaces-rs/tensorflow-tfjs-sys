use crate::tensors::Tensor;
use js_sys::Promise;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Creates a Tensor from an image.
    pub fn from_pixels(pixels: &JsValue, num_channels: usize) -> Tensor;

    /// Draws a Tensor of pixel values to a byte array or optionally a canvas.
    #[must_use]
    pub fn to_pixels(img: &Tensor, canvas: Option<&HtmlCanvasElement>) -> Promise;
}
