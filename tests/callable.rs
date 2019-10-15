mod tensor {
    use js_sys::Array;
    use tfjs_sys as tf;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn as_1d() {
        let value = Array::new();
        let dtype = Default::default();
        let tensor = tf::tensor1d(&value.into(), dtype);
        tensor.as_1d();
    }

    // FIXME
    // #[wasm_bindgen_test]
    // fn as_2d() {
    //     let value = {
    //         let res = Array::new();
    //         res.push(&1.into());
    //         res.push(&2.into());
    //         res.push(&3.into());
    //         res.push(&4.into());
    //         res
    //     };
    //     let shape: Option<&[usize]> = Some(&[2, 2]);
    //     let dtype = Default::default();
    //     let tensor = tf::tensor2d(&value.into(), shape, dtype);
    //     let rows = Default::default();
    //     let columns = Default::default();
    //     let tensor = tensor.as_2d(rows, columns);
    //     tensor.print(false);
    // }

    // FIXME
    // #[wasm_bindgen_test]
    // fn as_3d() {
    //     let value = {
    //         let res = Array::new();
    //         res.push(&1.into());
    //         res.push(&2.into());
    //         res.push(&3.into());
    //         res.push(&4.into());
    //         res
    //     };
    //     let shape: Option<&[usize]> = Some(&[2, 2, 1]);
    //     let dtype = Default::default();
    //     let tensor = tf::tensor3d(&value.into(), shape, dtype);
    //     let rows = Default::default();
    //     let columns = Default::default();
    //     let depth = Default::default();
    //     tensor.as_3d(rows, columns, depth);
    // }

    // FIXME
    // #[wasm_bindgen_test]
    // fn as_4d() {
    //     let value = {
    //         let res = Array::new();
    //         res.push(&1.into());
    //         res.push(&2.into());
    //         res.push(&3.into());
    //         res.push(&4.into());
    //         res
    //     };
    //     let shape: Option<&[usize]> = Some(&[1, 2, 2, 1]);
    //     let dtype = Default::default();
    //     let tensor = tf::tensor4d(&value.into(), shape, dtype);
    //     let rows = Default::default();
    //     let columns = Default::default();
    //     let depth = Default::default();
    //     let depth2 = Default::default();
    //     tensor.as_4d(rows, columns, depth, depth2);
    // }

    // FIXME
    // #[wasm_bindgen_test]
    // fn as_5d() {
    //     let value = {
    //         let res = Array::new();
    //         res.push(&1.into());
    //         res.push(&2.into());
    //         res.push(&3.into());
    //         res.push(&4.into());
    //         res.push(&5.into());
    //         res.push(&6.into());
    //         res.push(&7.into());
    //         res.push(&8.into());
    //         res
    //     };
    //     let shape: Option<&[usize]> = Some(&[1, 2, 2, 2, 1]);
    //     let dtype = Default::default();
    //     let tensor = tf::tensor5d(&value.into(), shape, dtype);
    //     let rows = Default::default();
    //     let columns = Default::default();
    //     let depth = Default::default();
    //     let depth2 = Default::default();
    //     let depth3 = Default::default();
    //     tensor.as_5d(rows, columns, depth, depth2, depth3);
    // }

    #[wasm_bindgen_test]
    fn as_scalar() {
        let value: u32 = Default::default();
        let dtype = Default::default();
        let tensor = tf::scalar(&value.into(), dtype);
        tensor.as_scalar();
    }

    #[wasm_bindgen_test]
    fn buffer() {
        let shape: &[usize] = &[];
        let dtype = Default::default();
        let values = Default::default();
        tf::buffer(&shape, dtype, values);
    }

    #[wasm_bindgen_test]
    fn eye() {
        let num_rows = Default::default();
        let num_columns = Default::default();
        let batch_shape = Default::default();
        let dtype = Default::default();
        tf::eye(num_rows, num_columns, batch_shape, dtype);
    }

    #[wasm_bindgen_test]
    fn print() {
        let value: u32 = Default::default();
        let dtype = Default::default();
        let tensor = tf::scalar(&value.into(), dtype);
        tensor.print(false);
    }

    #[wasm_bindgen_test]
    fn scalar() {
        let value: u32 = Default::default();
        let dtype = Default::default();
        tf::scalar(&value.into(), dtype);
    }

    #[wasm_bindgen_test]
    fn tensor() {
        let values = Array::new();
        let shape = Default::default();
        let dtype = Default::default();
        tf::tensor(&values, shape, dtype);
    }

    #[wasm_bindgen_test]
    fn tensor1d() {
        let values = Array::new();
        let dtype = Default::default();
        tf::tensor1d(&values, dtype);
    }

    #[wasm_bindgen_test]
    fn tensor2d() {
        let values = {
            let res = Array::new();
            res.push(&1.into());
            res.push(&2.into());
            res.push(&3.into());
            res.push(&4.into());
            res
        };
        let shape: Option<&[usize]> = Some(&[2, 2]);
        let dtype = Default::default();
        tf::tensor2d(&values, shape, dtype);
    }

    #[wasm_bindgen_test]
    fn tensor3d() {
        let values = {
            let res = Array::new();
            res.push(&1.into());
            res.push(&2.into());
            res.push(&3.into());
            res.push(&4.into());
            res
        };
        let shape: Option<&[usize]> = Some(&[2, 2, 1]);
        let dtype = Default::default();
        tf::tensor3d(&values, shape, dtype);
    }

    #[wasm_bindgen_test]
    fn tensor4d() {
        let values = {
            let res = Array::new();
            res.push(&1.into());
            res.push(&2.into());
            res.push(&3.into());
            res.push(&4.into());
            res
        };
        let shape: Option<&[usize]> = Some(&[1, 2, 2, 1]);
        let dtype = Default::default();
        tf::tensor4d(&values, shape, dtype);
    }

    #[wasm_bindgen_test]
    fn tensor5d() {
        let values = {
            let res = Array::new();
            res.push(&1.into());
            res.push(&2.into());
            res.push(&3.into());
            res.push(&4.into());
            res.push(&5.into());
            res.push(&6.into());
            res.push(&7.into());
            res.push(&8.into());
            res
        };
        let shape: Option<&[usize]> = Some(&[1, 2, 2, 2, 1]);
        let dtype = Default::default();
        tf::tensor5d(&values, shape, dtype);
    }

    #[wasm_bindgen_test]
    fn tensor6d() {
        let values = {
            let res = Array::new();
            res.push(&1.into());
            res.push(&2.into());
            res.push(&3.into());
            res.push(&4.into());
            res.push(&5.into());
            res.push(&6.into());
            res.push(&7.into());
            res.push(&8.into());
            res
        };
        let shape: Option<&[usize]> = Some(&[1, 1, 2, 2, 2, 1]);
        let dtype = Default::default();
        tf::tensor6d(&values, shape, dtype);
    }
}

mod tensor_buffer {
    use tfjs_sys as tf;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn buffer() {
        let shape = [];
        let dtype = Default::default();
        let values = Default::default();
        let buffer = tf::buffer(&shape, dtype, values);
        buffer.to_tensor().print(false);
    }

    #[wasm_bindgen_test]
    fn set() {
        let shape = [2, 2];
        let dtype = Default::default();
        let values = Default::default();
        let buffer = tf::buffer(&shape, dtype, values);
        buffer.set(&3.into(), &[0, 0]);
        buffer.set(&5.into(), &[1, 0]);
        buffer.to_tensor().print(false);
    }
}

mod variable {
    use js_sys::Array;
    use tfjs_sys as tf;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn assign() {
        let old = {
            let values = {
                let res = Array::new();
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
                let res = Array::new();
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
