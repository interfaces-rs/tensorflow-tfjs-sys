use crate::{data::Dataset, layers::Layer};
use js_sys::{Function, JsString, Object, Promise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A GraphModel is a directed, acyclic graph of built from SavedModel GraphDef and allows
    /// inference exeuction.
    pub type GraphModel;

    /// Execute the inference for the input tensors.
    #[wasm_bindgen(method)]
    pub fn predict(this: &GraphModel, inputs: &JsValue, config: Option<&Object>) -> JsValue;

    /// Executes inference for the model for given input tensors.
    #[wasm_bindgen(method)]
    pub fn execute(this: &GraphModel, inputs: &JsValue, outputs: Option<&Object>) -> JsValue;

    /// Executes inference for the model for given input tensors in async fashion, use this method
    /// when your model contains control flow ops.
    #[wasm_bindgen(method, js_name = "executeAsync")]
    pub fn execute_async(this: &GraphModel, inputs: &JsValue, outputs: Option<&Object>) -> Promise;

    /// Releases the memory used by the weight tensors.
    #[wasm_bindgen(method)]
    pub fn dispose(this: &GraphModel);
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A LayersModel is a directed, acyclic graph of Layers plus methods for training,
    /// evaluation, prediction and saving.
    pub type LayersModel;

    /// Configures and prepares the model for training and evaluation. Compiling outfits the model
    /// with an optimizer, loss, and/or metrics. Calling fit or evaluate on an un-compiled model
    /// will throw an error.
    #[wasm_bindgen(method)]
    pub fn compile(this: &LayersModel, args: &Object);

    /// Returns the loss value & metrics values for the model in test mode.
    #[wasm_bindgen(method)]
    pub fn evaluate(this: &LayersModel, x: &JsValue, y: &JsValue, args: Option<&Object>) -> JsValue; // FIXME

    /// Evaluate model using a dataset object.
    #[wasm_bindgen(method, js_name = "evaluateDataset")]
    #[must_use]
    pub fn evaluate_dataset(this: &LayersModel, dataset: &Dataset, args: Option<&Object>) -> Promise;

    /// Trains the model for a fixed number of epochs (iterations on a dataset).
    #[wasm_bindgen(method)]
    #[must_use]
    pub fn fit(this: &LayersModel, x: &JsValue, y: &JsValue, args: Option<&Object>) -> Promise;

    /// Trains the model using a dataset object.
    #[wasm_bindgen(method, js_name = "fitDataset")]
    #[must_use]
    pub fn fit_dataset(this: &LayersModel, dataset: &Dataset, args: &Object) -> Promise;

    /// Retrieves a layer based on either its name (unique) or index.
    #[wasm_bindgen(method, js_name = "getLayer")]
    pub fn get_layer(this: &LayersModel, name: Option<&JsString>, index: Option<usize>) -> Layer;

    /// Generates output predictions for the input samples.
    #[wasm_bindgen(method)]
    pub fn predict(this: &LayersModel, x: &JsValue, args: Option<&Object>) -> JsValue;

    /// Returns predictions for a single batch of samples.
    #[wasm_bindgen(method, js_name = "predictOnBatch")]
    pub fn predict_on_batch(this: &LayersModel, x: &JsValue) -> JsValue;

    /// Save the configuration and/or weights of the LayersModel.
    #[wasm_bindgen(method)]
    #[must_use]
    pub fn save(this: &LayersModel, url: &JsString, config: Option<&Object>) -> Promise;

    /// Print a text summary of the model's layers.
    #[wasm_bindgen(method)]
    pub fn summary(this: &LayersModel, line_length: Option<u32>, position: Option<&[u32]>, print_fn: &Function);

    /// Runs a single gradient update on a single batch of data.
    #[wasm_bindgen(method, js_name = "trainOnBatch")]
    #[must_use]
    pub fn train_on_batch(this: &LayersModel, x: &JsValue, y: &JsValue) -> Promise;
}
