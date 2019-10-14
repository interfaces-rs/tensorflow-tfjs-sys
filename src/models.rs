use crate::{data::Dataset, layers::Layer, tensors::SymbolicTensor, OpMapper};
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

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// A model with a stack of layers, feeding linearly from one to the next.
    #[wasm_bindgen(extends = LayersModel)]
    pub type Sequential;

    /// Adds a layer instance on top of the layer stack.
    #[wasm_bindgen(method)]
    pub fn add(this: &Sequential, layer: &Layer);

    /// Print a text summary of the Sequential model's layers.
    #[wasm_bindgen(method)]
    pub fn summary(this: &Sequential, line_length: Option<u32>, positions: &[u32], print_fn: &Function);

    /// Returns the loss value & metrics values for the model in test mode.
    #[wasm_bindgen(method)]
    pub fn evaluate(this: &Sequential, x: &JsValue, y: &JsValue, args: Option<&Object>) -> JsValue;

    /// Evaluate model using a dataset object.
    #[wasm_bindgen(method, js_name = "evaluateDataset")]
    pub fn evaluate_dataset(this: &Sequential, dataset: &Dataset, args: &Object) -> Promise;

    /// Generates output predictions for the input samples.
    #[wasm_bindgen(method)]
    pub fn predict(this: &Sequential, x: &JsValue, args: Option<&Object>) -> JsValue;

    /// Trains the model for a fixed number of epochs (iterations on a dataset).
    #[wasm_bindgen(method)]
    pub fn fit(this: &Sequential, x: &JsValue, y: &JsValue, args: Option<&Object>) -> Promise;

    /// Trains the model using a dataset object.
    #[wasm_bindgen(method, js_name = "fitDataset")]
    pub fn fit_dataset(this: &Sequential, dataset: &Dataset, args: &Object) -> Promise;

    /// Runs a single gradient update on a single batch of data.
    #[wasm_bindgen(method, js_name = "trainOnBatch")]
    pub fn train_on_batch(this: &Sequential, x: &JsValue, y: &JsValue) -> Promise;
}

pub mod io {
    use crate::IOHandler;
    use js_sys::{Array, Object, Promise};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(module = "@tensorflow/tfjs")]
    extern {
        /// Creates an IOHandler that triggers file downloads from the browser.
        #[wasm_bindgen(js_name = "browserDownloads")]
        pub fn browser_downloads(file_name_prefix: Option<&str>) -> IOHandler;

        /// Creates an IOHandler that loads model artifacts from user-selected files.
        #[wasm_bindgen(js_name = "browserFiles")]
        pub fn browser_files(files: &Array) -> IOHandler;

        /// Copy a model from one URL to another.
        pub fn copy_model(src: &str, dst: &str) -> Promise;

        /// Creates an IOHandler subtype that sends model artifacts to HTTP server.
        pub fn http(path: &str, load_options: Option<&Object>) -> IOHandler;

        /// List all models stored in registered storage mediums.
        pub fn list_models() -> Promise;

        /// Move a model from one URL to another.
        pub fn move_model(src: &str, dst: &str) -> Promise;

        /// Remove a model specified by URL from a reigstered storage medium.
        pub fn remove_model(url: &str) -> Promise;
    }
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Creates a Sequential model. A sequential model is any model where the outputs of one
    /// layer are the inputs to the next layer, i.e. the model topology is a simple 'stack' of
    /// layers, with no branching or skipping.
    pub fn sequential(config: Option<&Object>) -> Sequential;

    /// A model is a data structure that consists of Layers and defines inputs and outputs.
    pub fn model(args: &Object) -> LayersModel;

    /// Load a graph model given a URL to the model definition.
    #[wasm_bindgen(js_name = "loadGraphModel")]
    pub fn load_graph_model(model_url: &JsValue, options: Option<&Object>) -> Promise;

    /// Load a model composed of Layer objects, including its topology and optionally weights. See
    /// the Tutorial named "How to import a Keras Model" for usage examples.
    pub fn load_layers_model(path_or_io_handler: &JsValue, options: Option<&Object>) -> Promise;

    /// Used to instantiate an input to a model as a SymbolicTensor.
    pub fn input(config: &Object) -> SymbolicTensor;

    /// Register a class with the serialization map of TensorFlow.js.
    #[wasm_bindgen(js_name = "registerClass")]
    pub fn register_class(cls: &JsValue);

    /// Deregister the Op for graph model executor.
    pub fn deregister_op(name: &str);

    /// Retrieve the OpMapper object for the registered op.
    pub fn get_registered_op(name: &str) -> OpMapper;

    /// Register an Op for graph model executor. This allow you to register TensorFlow custom op or
    /// override existing op.
    pub fn registered_op(name: &str, op_func: &Object) -> OpMapper;
}
