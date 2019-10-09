use tfjs_sys as tf;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
async fn scalar_callable() {
    let value = 0u32.into();
    let dtype = Default::default();
    tf::scalar(&value, dtype);
}

#[wasm_bindgen_test]
async fn tensor_callable() {
    use js_sys::Array;
    let values = Array::new();
    let shape = Default::default();
    let dtype = Default::default();
    tf::tensor(&values, shape, dtype);
}
