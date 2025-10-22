use crate::core::traits::TensorLike;
use ndarray::{ArrayBase, Dim, Data, IntoDimension, Array2, ArrayD};
use crate::errors::DataError;

impl TensorLike for Array2<f64> {
    type Elem = f64;

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_array(&self) -> Result<ArrayD<Self::Elem>, DataError> {
        Ok(self.clone().into_dyn())
    }

    fn type_name(&self) -> &'static str {
        "NdArray2D<f64>"
    }
}