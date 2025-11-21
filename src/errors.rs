use thiserror::Error;
use ndarray_stats::errors::MinMaxError;

impl From<MinMaxError> for DataError {
    fn from(error: MinMaxError) -> Self {
        DataError::MinMaxComputationError(error.to_string())
    }
}

#[derive(Debug, Error)]
pub enum DataError {
    #[error("Error of dimension: X has {0} and y has {1}")]
    //DimensionMismatch(#[from] ndarray::ShapeError),
    DimensionMismatch(usize, usize),
    #[error("Error:{0}")]
    CalculationError(String),
    #[error("The data cannot be empty")]
    EmptyData,
    #[error("The target is empty")]
    EmptyTarget,
    #[error("Presence of Nan values")]
    NaNData,
    #[error("The model is not fitted")]
    NoteFittedModel,
    #[error("The available metrics is Manhattan and Euclidean")]
    InvalidMetric,
    #[error("Invalid data")]
    InvalidData,
    #[error("{0}")]
    MinMaxComputationError(String)
}
