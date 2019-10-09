use tfjs_sys as tf;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
async fn print_callable() {
    let value: u32 = Default::default();
    let dtype = Default::default();
    let tensor = tf::scalar(&value.into(), dtype);
    tensor.print(false);
}

#[wasm_bindgen_test]
async fn scalar_callable() {
    let value: u32 = Default::default();
    let dtype = Default::default();
    let tensor = tf::scalar(&value.into(), dtype);
    tensor.print(false);
}

#[wasm_bindgen_test]
async fn tensor_callable() {
    use js_sys::Array;
    let values = Array::new();
    let shape = Default::default();
    let dtype = Default::default();
    let tensor = tf::tensor(&values, shape, dtype);
    tensor.print(false);
}
