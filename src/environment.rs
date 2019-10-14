#[wasm_bindgen(module = "@tensorflow/tfjs")]
extern {
    /// Dispose all variables kept in backend engine.
    #[wasm_bindgen(js_name = "disposeVariables")]
    pub fn dispose_variables();

    /// Enables debug mode which will log information about all executed kernels: the elapsed time of the kernel execution, as well as the rank, shape, and size of the output tensor.
    #[wasm_bindgen(js_name = "enableDebugMode")]
    pub fn enable_debug_mode();

    /// Enables production mode which disables correctness checks in favor of performance.
    #[wasm_bindgen(js_name = "enableProdMode")]
    pub fn enable_prod_mode();
}
