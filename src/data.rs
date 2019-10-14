use crate::DatasetContainer;
use js_sys::{Array, Function, Object, Promise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Represents a potentially large collection of delimited text records.
    #[wasm_bindgen(extends = Dataset)]
    pub type CSVDataset;

    /// Returns column names of the csv dataset.
    #[wasm_bindgen(method, js_name = "columnNames")]
    pub fn column_names(this: &CSVDataset) -> Object;
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Represents a potentially large list of independent data elements (typically 'samples' or
    /// 'examples').
    pub type Dataset;

    /// Groups elements into batches.
    #[wasm_bindgen(method)]
    pub fn batch(this: &Dataset, batch_size: usize, small_last_batch: Option<bool>) -> Dataset;

    /// Concatenates this Dataset with another.
    #[wasm_bindgen(method)]
    pub fn concatenate(this: &Dataset, that: &Dataset) -> Dataset;

    /// Filters this dataset according to predicate.
    #[wasm_bindgen(method)]
    pub fn filter(this: &Dataset, predicate: &Function) -> Dataset;

    /// Apply a function to every element of the dataset.
    #[wasm_bindgen(method, js_name = "forEachAsync")]
    #[must_use]
    pub fn for_each_async(this: &Dataset, transform: &Function) -> Promise;

    /// Maps this dataset through a 1-to-1 transform.
    #[wasm_bindgen(method)]
    pub fn map(this: &Dataset, transform: &Function) -> Dataset;

    /// Maps this dataset through an async 1-to-1 transform.
    #[wasm_bindgen(method, js_name = "mapAsync")]
    pub fn map_async(this: &Dataset, transform: &Function) -> Dataset;

    /// Creates a Dataset that prefetches elements from this dataset.
    #[wasm_bindgen(method)]
    pub fn prefetch(this: &Dataset, buffer_size: usize) -> Dataset;

    /// Repeats this dataset count times.
    #[wasm_bindgen(method)]
    pub fn repeat(this: &Dataset, count: Option<usize>) -> Dataset;

    /// Creates a Dataset that skips count initial elements from this dataset.
    #[wasm_bindgen(method)]
    pub fn skip(this: &Dataset, count: Option<usize>) -> Dataset;

    /// Pseudorandomly shuffles the elements of this dataset. This is done in a streaming manner, by
    /// sampling from a given number of prefetched elements.
    #[wasm_bindgen(method)]
    pub fn shuffle(
        this: &Dataset,
        buffer_size: usize,
        seed: Option<&str>,
        reshuffle_each_iteration: Option<bool>,
    ) -> Dataset;

    /// Creates a Dataset with at most count initial elements from this dataset.
    #[wasm_bindgen(method)]
    pub fn take(this: &Dataset, count: Option<usize>) -> Dataset;

    /// Collect all elements of this dataset into an array.
    #[wasm_bindgen(method, js_name = "toArray")]
    #[must_use]
    pub fn to_array(this: &Dataset) -> Promise;
}

#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Create a Dataset from an array of elements.
    pub fn array(items: &Array) -> Dataset;

    /// Create a CSVDataset by reading and decoding CSV file(s) from provided URL or local path if
    /// it's in Node environment.
    pub fn csv(source: &JsValue, csv_config: Option<&Object>) -> CSVDataset;

    /// Create a Dataset that produces each element from provided JavaScript generator.
    pub fn generator(fun: &Function) -> Dataset;

    /// Create an iterator that generate frequency-domain spectrogram Tensors from microphone audio
    /// stream with browser's native FFT. This API only works in browser environment when the device
    /// has microphone.
    #[must_use]
    pub fn microphone(config: Option<&Object>) -> Promise;

    /// Create an iterator that generate Tensors from webcam video stream. This API only works in
    /// Browser environment when the device has webcam.
    // FIXME: use web-sys for elem type
    #[must_use]
    pub fn webcam(elem: &JsValue, config: Option<&Object>) -> Promise;

    /// Create a Dataset by zipping together an array, dict, or nested structure of Datasets (and
    /// perhaps additional constants). The underlying datasets must provide elements in a consistent
    /// order such that they correspond.
    pub fn zip(datasets: &DatasetContainer) -> Dataset;
}
