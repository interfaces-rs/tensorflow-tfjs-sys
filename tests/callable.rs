mod tensor {
    use tfjs_sys as tf;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    async fn print() {
        let value: u32 = Default::default();
        let dtype = Default::default();
        let tensor = tf::scalar(&value.into(), dtype);
        tensor.print(false);
    }

    #[wasm_bindgen_test]
    async fn scalar() {
        let value: u32 = Default::default();
        let dtype = Default::default();
        let tensor = tf::scalar(&value.into(), dtype);
        tensor.print(false);
    }

    #[wasm_bindgen_test]
    async fn tensor() {
        use js_sys::Array;
        let values = Array::new();
        let shape = Default::default();
        let dtype = Default::default();
        let tensor = tf::tensor(&values, shape, dtype);
        tensor.print(false);
    }
}

mod variable {
    use tfjs_sys as tf;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn assign() {
        let old = {
            let values = {
                let res = js_sys::Array::new();
                res.push(&0u32.into());
                res.push(&1u32.into());
                res
            };
            let shape = Default::default();
            let dtype = Default::default();
            tf::tensor(&values, shape, dtype)
        };
        let new = {
            let values = {
                let res = js_sys::Array::new();
                res.push(&2u32.into());
                res.push(&3u32.into());
                res
            };
            let shape = Default::default();
            let dtype = Default::default();
            tf::tensor(&values, shape, dtype)
        };
        let variable = tf::Variable::new(&old);
        variable.assign(&new);
    }
}
