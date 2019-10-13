use js_sys::Object;
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
}
