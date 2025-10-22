use ndarray::{Array2, ArrayD, Axis};
use crate::errors::DataError;
use ndarray_rand::rand_distr::num_traits;

// Definition of the tensorlike trait
pub trait TensorLike {
    type Elem: Clone + PartialEq;

    /// Returns the shape of the tensor
    fn shape(&self) -> &[usize];

    fn len(&self) -> usize;

    /// If the tensor is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the tensor into an ArrayD (for interoperability)
    fn to_array(&self) -> Result<ArrayD<Self::Elem>, DataError>;

    /// Name of the type for debug
    fn type_name(&self) -> &'static str;
}

/// Trait for statistics operations
pub trait  StatisticsOps: TensorLike
where
    Self::Elem: num_traits::Float,
{
    // All the operations returns an ArrayD
    fn mean(&self) -> Result<ArrayD<Self::Elem>, DataError>;
    fn std(&self) -> Result<ArrayD<Self::Elem>, DataError>;
    fn min(&self) -> Result<ArrayD<Self::Elem>, DataError>;
    fn max(&self) -> Result<ArrayD<Self::Elem>, DataError>;
    fn median(&self) -> Result<ArrayD<Self::Elem>, DataError>;
    fn first_quantile(&self) -> Result<ArrayD<Self::Elem>, DataError>;
    fn third_quantile(&self) -> Result<ArrayD<Self::Elem>, DataError>;
}

/// Trait for matrixial tensors (linear agelbra operations)
pub trait MatrixOps: TensorLike {
    fn transpose(&self) -> Result<Box<dyn TensorLike<Elem=Self::Elem>>, DataError>;
    fn matmul(&self, other: &dyn TensorLike<Elem = Self::Elem>) -> Result<Box<dyn TensorLike<Elem=Self::Elem>>, DataError>;
}