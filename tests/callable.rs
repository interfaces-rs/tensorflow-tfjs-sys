mod helper {
    use js_sys::Array;
    use tfjs_sys as tf;

    pub fn tensor() -> tf::Tensor {
        let value = Array::new();
        let dtype = Default::default();
        tf::tensor1d(&value.into(), dtype)
    }
}

mod tensor {
    mod functions {
        use super::super::helper;
        use js_sys::Array;
        use tfjs_sys as tf;
        use wasm_bindgen_futures::JsFuture;
        use wasm_bindgen_test::*;

        #[wasm_bindgen_test]
        fn add() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            tf::add(&lhs, &rhs);
        }

        #[wasm_bindgen_test]
        fn add_n() {
            let tensors = {
                let res = Array::new();
                res.push(&helper::tensor());
                res
            };
            tf::add_n(&tensors);
        }

        // FIXME
        // #[wasm_bindgen_test]
        // fn batch_space_to_nd() {
        //     let x = {
        //         let values = {
        //             let res = Array::new();
        //             res.push(&1.into());
        //             res.push(&2.into());
        //             res.push(&3.into());
        //             res.push(&4.into());
        //             res
        //         };
        //         let shape: Option<&[i32]> = Some(&[4, 1, 1, 1]);
        //         let dtype = Default::default();
        //         tf::tensor4d(&values, shape, dtype)
        //     };
        //     let block_shape: &[usize] = &[2, 2];
        //     let crops = {
        //         let res = Array::new();
        //         res.push(&{
        //             let res = Array::new();
        //             res.push(&0.into());
        //             res.push(&0.into());
        //             res.into()
        //         });
        //         res.push(&{
        //             let res = Array::new();
        //             res.push(&0.into());
        //             res.push(&0.into());
        //             res.into()
        //         });
        //         res
        //     };
        //     tf::batch_to_space_nd(&x, block_shape, &crops);
        // }

        #[wasm_bindgen_test]
        fn buffer() {
            let shape: &[i32] = &[];
            let dtype = Default::default();
            let values = Default::default();
            tf::buffer(&shape, dtype, values);
        }

        #[wasm_bindgen_test]
        fn cast() {
            let value = Array::new();
            let dtype = "float32";
            let tensor = tf::tensor1d(&value.into(), Some(dtype));
            tf::cast(&tensor, dtype);
        }

        #[wasm_bindgen_test]
        fn clone() {
            let tensor = helper::tensor();
            tf::clone(&tensor);
        }

        #[wasm_bindgen_test]
        fn complex() {
            let real = Default::default();
            let imag = Default::default();
            tf::complex(real, imag);
        }

        #[wasm_bindgen_test]
        fn concat() {
            let tensors = Array::new();
            tensors.push(&helper::tensor());
            tensors.push(&helper::tensor());
            let axis = Default::default();
            tf::concat(&tensors, axis);
        }

        #[wasm_bindgen_test]
        fn depth_to_space_nd() {
            let x = {
                let values = {
                    let res = Array::new();
                    res.push(&1.into());
                    res.push(&2.into());
                    res.push(&3.into());
                    res.push(&4.into());
                    res
                };
                let shape: Option<&[i32]> = Some(&[1, 1, 1, 4]);
                let dtype = Default::default();
                tf::tensor4d(&values, shape, dtype)
            };
            let block_size = 2;
            let data_format = Default::default();
            tf::depth_to_space(&x, block_size, data_format);
        }

        #[wasm_bindgen_test]
        fn div() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            tf::div(&lhs, &rhs);
        }

        // FIXME
        // #[wasm_bindgen_test]
        // fn expand_dims() {
        //     let x = {
        //         let values = {
        //             let res = Array::new();
        //             res.push(&1.into());
        //             res.push(&2.into());
        //             res.push(&3.into());
        //             res.push(&4.into());
        //             res
        //         };
        //         let dtype = Default::default();
        //         tf::tensor1d(&values, dtype)
        //     };
        //     let axis = Some(1);
        //     tf::expand_dims(&x, axis);
        // }

        #[wasm_bindgen_test]
        fn eye() {
            let num_rows = Default::default();
            let num_columns = Default::default();
            let batch_shape = Default::default();
            let dtype = Default::default();
            tf::eye(num_rows, num_columns, batch_shape, dtype);
        }

        #[wasm_bindgen_test]
        fn fill() {
            let shape: &[i32] = &[2, 2];
            let value = 4;
            let dtype = Default::default();
            tf::fill(shape, value, dtype);
        }

        #[wasm_bindgen_test]
        fn floor_div() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            tf::floor_div(&lhs, &rhs);
        }

        #[wasm_bindgen_test]
        fn gather() {
            let x = helper::tensor();
            let indices = helper::tensor().to_int();
            let axis = Default::default();
            tf::gather(&x, &indices, axis);
        }

        #[wasm_bindgen_test]
        fn imag() {
            let real = Default::default();
            let imag = Default::default();
            let complex = tf::complex(real, imag);
            tf::imag(&complex);
        }

        #[wasm_bindgen_test]
        fn linspace() {
            let start = 0;
            let stop = 9;
            let num = 10;
            tf::linspace(start, stop, num);
        }

        #[wasm_bindgen_test]
        fn maximum() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            tf::maximum(&lhs, &rhs);
        }

        #[wasm_bindgen_test]
        fn minimum() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            tf::minimum(&lhs, &rhs);
        }

        #[wasm_bindgen_test]
        fn mul() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            tf::mul(&lhs, &rhs);
        }

        #[wasm_bindgen_test]
        fn multinomial() {
            let values = {
                let res = Array::new();
                res.push(&0.75.into());
                res.push(&0.25.into());
                res
            };
            let shape = Default::default();
            let dtype = Default::default();
            let probs = tf::tensor(&values, shape, dtype);
            let seed = Default::default();
            let normalized = Default::default();
            tf::multinomial(&probs, 3, seed, normalized);
        }

        #[wasm_bindgen_test]
        fn one_hot() {
            let indices = {
                let values = {
                    let res = Array::new();
                    res.push(&0.into());
                    res.push(&1.into());
                    res
                };
                let dtype = Some("int32");
                tf::tensor1d(&values, dtype)
            };
            let depth = 3;
            let on_value = Default::default();
            let off_value = Default::default();
            tf::one_hot(&indices, depth, on_value, off_value);
        }

        #[wasm_bindgen_test]
        fn ones() {
            let shape: &[i32] = &[2, 2];
            let dtype = Default::default();
            tf::ones(shape, dtype);
        }

        #[wasm_bindgen_test]
        fn ones_like() {
            let value = Array::new();
            let dtype = Default::default();
            let tensor = tf::tensor1d(&value.into(), dtype);
            tf::ones_like(&tensor);
        }

        #[wasm_bindgen_test]
        fn pad() {
            let x = helper::tensor();
            let padding = Array::new();
            let constant_value = Default::default();
            tf::pad(&x, &padding, constant_value);
        }

        #[wasm_bindgen_test]
        fn pow() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            tf::pow(&lhs, &rhs);
        }

        #[wasm_bindgen_test]
        fn print() {
            let tensor = helper::tensor();
            let verbose = Default::default();
            tf::print(&tensor, verbose);
        }

        #[wasm_bindgen_test]
        fn modulus() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            tf::modulus(&lhs, &rhs);
        }

        #[wasm_bindgen_test]
        fn random_normal() {
            let shape: &[i32] = &[2, 2];
            let mean = Default::default();
            let std_dev = Default::default();
            let dtype = Default::default();
            let seed = Default::default();
            tf::random_normal(shape, mean, std_dev, dtype, seed);
        }

        #[wasm_bindgen_test]
        fn random_uniform() {
            let shape: &[i32] = &[2, 2];
            let minval = Default::default();
            let maxval = Default::default();
            let dtype = Default::default();
            let seed = Default::default();
            tf::random_normal(shape, minval, maxval, dtype, seed);
        }

        #[wasm_bindgen_test]
        fn range() {
            let start = 0;
            let stop = 9;
            let step = Default::default();
            let dtype = Default::default();
            tf::range(start, stop, step, dtype);
        }

        #[wasm_bindgen_test]
        fn real() {
            let real = Default::default();
            let imag = Default::default();
            let complex = tf::complex(real, imag);
            tf::real(&complex);
        }

        #[wasm_bindgen_test]
        fn reshape() {
            let values = {
                let res = Array::new();
                res.push(&1.into());
                res.push(&2.into());
                res.push(&3.into());
                res.push(&4.into());
                res
            };
            let dtype = Default::default();
            let tensor = tf::tensor1d(&values, dtype);
            tf::reshape(&tensor, &[2, 2]);
        }

        #[wasm_bindgen_test]
        fn reverse() {
            let x = helper::tensor();
            let axis = Default::default();
            tf::reverse(&x, axis);
        }

        #[wasm_bindgen_test]
        fn scalar() {
            let value = Default::default();
            let dtype = Default::default();
            tf::scalar(value, dtype);
        }

        #[wasm_bindgen_test]
        async fn set_diff_1d_async() {
            let lhs = helper::tensor();
            let rhs = helper::tensor();
            let promise = tf::set_diff_1d_async(&lhs, &rhs);
            JsFuture::from(promise).await.unwrap();
        }

        #[wasm_bindgen_test]
        fn squeeze() {
            let values = {
                let res = Array::new();
                res.push(&1.into());
                res.push(&2.into());
                res.push(&3.into());
                res.push(&4.into());
                res
            };
            let shape: Option<&[i32]> = Some(&[1, 1, 4]);
            let dtype = Default::default();
            let tensor = tf::tensor(&values, shape, dtype);
            let axis = Default::default();
            tf::squeeze(&tensor, axis);
        }

        #[wasm_bindgen_test]
        fn stack() {
            let tensors = Array::new();
            tensors.push(&helper::tensor());
            let axis = Default::default();
            tf::stack(&tensors, axis);
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
            let shape: Option<&[i32]> = Some(&[2, 2]);
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
            let shape: Option<&[i32]> = Some(&[2, 2, 1]);
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
            let shape: Option<&[i32]> = Some(&[1, 2, 2, 1]);
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
            let shape: Option<&[i32]> = Some(&[1, 2, 2, 2, 1]);
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
            let shape: Option<&[i32]> = Some(&[1, 1, 2, 2, 2, 1]);
            let dtype = Default::default();
            tf::tensor6d(&values, shape, dtype);
        }

        #[wasm_bindgen_test]
        fn truncated_normal() {
            let shape: &[i32] = &[2, 2];
            let mean = Default::default();
            let std_dev = Default::default();
            let dtype = Default::default();
            let seed = Default::default();
            tf::truncated_normal(shape, mean, std_dev, dtype, seed);
        }

        #[wasm_bindgen_test]
        fn unstack() {
            let values = {
                let res = Array::new();
                res.push(&1.into());
                res.push(&2.into());
                res.push(&3.into());
                res.push(&4.into());
                res
            };
            let shape: Option<&[i32]> = Some(&[2, 2]);
            let dtype = Default::default();
            let x = tf::tensor2d(&values, shape, dtype);
            let axis = Default::default();
            tf::unstack(&x, axis);
        }

        #[wasm_bindgen_test]
        fn variable() {
            let value = Array::new();
            let dtype = Default::default();
            let tensor = tf::tensor1d(&value.into(), dtype);
            let trainable = Default::default();
            let name = Default::default();
            let dtype = Default::default();
            tf::variable(&tensor, trainable, name, dtype);
        }

        #[wasm_bindgen_test]
        fn zeros() {
            let shape: &[i32] = &[2, 2];
            let dtype = Default::default();
            tf::zeros(shape, dtype);
        }

        #[wasm_bindgen_test]
        fn zeros_like() {
            let tensor = helper::tensor();
            tf::zeros_like(&tensor);
        }
    }

    mod methods {
        use super::super::helper;
        use js_sys::Array;
        use tfjs_sys as tf;
        use wasm_bindgen_futures::JsFuture;
        use wasm_bindgen_test::*;

        #[wasm_bindgen_test]
        fn abs() {
            let this = helper::tensor();
            this.abs();
        }

        #[wasm_bindgen_test]
        fn as_1d() {
            let value = Array::new();
            let dtype = Default::default();
            let tensor = tf::tensor1d(&value.into(), dtype);
            tensor.as_1d();
        }

        #[wasm_bindgen_test]
        fn as_scalar() {
            let value = Default::default();
            let dtype = Default::default();
            let tensor = tf::scalar(value, dtype);
            tensor.as_scalar();
        }

        #[wasm_bindgen_test]
        fn as_type() {
            let value = Default::default();
            let dtype = Some("int32");
            let tensor = tf::scalar(value, dtype);
            tensor.as_type("float32");
        }

        #[wasm_bindgen_test]
        async fn array() {
            let tensor = helper::tensor();
            let promise = tensor.array();
            JsFuture::from(promise).await.unwrap();
        }

        #[wasm_bindgen_test]
        fn array_sync() {
            let tensor = helper::tensor();
            tensor.array_sync();
        }

        #[wasm_bindgen_test]
        async fn buffer() {
            let tensor = helper::tensor();
            let promise = tensor.buffer();
            JsFuture::from(promise).await.unwrap();
        }

        #[wasm_bindgen_test]
        fn buffer_sync() {
            let tensor = helper::tensor();
            tensor.buffer_sync();
        }

        #[wasm_bindgen_test]
        fn clone() {
            let tensor = helper::tensor();
            tensor.clone();
        }

        #[wasm_bindgen_test]
        fn cumsum() {
            let tensor = helper::tensor();
            let axis = Default::default();
            let exclusive = Default::default();
            let reverse = Default::default();
            tensor.cumsum(axis, exclusive, reverse);
        }

        #[wasm_bindgen_test]
        async fn data() {
            let tensor = helper::tensor();
            let promise = tensor.data();
            JsFuture::from(promise).await.unwrap();
        }

        #[wasm_bindgen_test]
        fn data_sync() {
            let tensor = helper::tensor();
            tensor.data_sync();
        }

        #[wasm_bindgen_test]
        fn dispose() {
            let tensor = helper::tensor();
            tensor.dispose();
        }

        #[wasm_bindgen_test]
        fn expand_dims() {
            let tensor = helper::tensor();
            let axis = Default::default();
            tensor.expand_dims(axis);
        }

        #[wasm_bindgen_test]
        fn flatten() {
            let tensor = helper::tensor();
            tensor.flatten();
        }

        #[wasm_bindgen_test]
        fn print() {
            let value = Default::default();
            let dtype = Default::default();
            let tensor = tf::scalar(value, dtype);
            tensor.print(Some(false));
        }

        #[wasm_bindgen_test]
        fn reshape() {
            let values = {
                let res = Array::new();
                res.push(&1.into());
                res.push(&2.into());
                res.push(&3.into());
                res.push(&4.into());
                res
            };
            let dtype = Default::default();
            let tensor = tf::tensor1d(&values, dtype);
            tensor.reshape(&[2, 2]);
        }

        #[wasm_bindgen_test]
        fn reshape_as() {
            let this = {
                let values = {
                    let res = Array::new();
                    res.push(&1.into());
                    res.push(&2.into());
                    res
                };
                let dtype = Default::default();
                tf::tensor1d(&values, dtype)
            };
            let that = {
                let values = {
                    let res = Array::new();
                    res.push(&2.into());
                    res.push(&2.into());
                    res
                };
                let dtype = Default::default();
                tf::tensor1d(&values, dtype)
            };
            this.reshape_as(&that);
        }

        #[wasm_bindgen_test]
        fn squeeze() {
            let values = {
                let res = Array::new();
                res.push(&1.into());
                res.push(&2.into());
                res.push(&3.into());
                res.push(&4.into());
                res
            };
            let shape: Option<&[i32]> = Some(&[1, 1, 4]);
            let dtype = Default::default();
            let tensor = tf::tensor(&values, shape, dtype);
            let axis = Default::default();
            tensor.squeeze(axis);
        }

        #[wasm_bindgen_test]
        fn to_bool() {
            let tensor = helper::tensor();
            tensor.to_bool();
        }

        #[wasm_bindgen_test]
        fn to_float() {
            let tensor = helper::tensor();
            tensor.to_float();
        }

        #[wasm_bindgen_test]
        fn to_int() {
            let tensor = helper::tensor();
            tensor.to_int();
        }

        #[wasm_bindgen_test]
        fn to_string() {
            let tensor = helper::tensor();
            tensor.to_string();
        }
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
        tf::buffer(&shape, dtype, values);
    }

    #[wasm_bindgen_test]
    fn get() {
        let shape = [2, 2];
        let dtype = Default::default();
        let values = Default::default();
        let buffer = tf::buffer(&shape, dtype, values);
        buffer.get(&[0, 0]);
    }

    #[wasm_bindgen_test]
    fn set() {
        let shape = [2, 2];
        let dtype = Default::default();
        let values = Default::default();
        let buffer = tf::buffer(&shape, dtype, values);
        buffer.set(&3.into(), &[0, 0]);
        buffer.set(&5.into(), &[1, 0]);
    }

    #[wasm_bindgen_test]
    fn to_tensor() {
        let shape = [];
        let dtype = Default::default();
        let values = Default::default();
        let buffer = tf::buffer(&shape, dtype, values);
        buffer.to_tensor();
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
