use thiserror::Error;


#[derive(Debug, Error)]
pub enum DataError {
    #[error("Error of dimension {0}")]
    DimensionMismatch(String),
    #[error("Error:{0}")]
    CalculationError(String),
    #[error("The data cannot be empty")]
    EmptyData,
    #[error("Ndarray error: {0}")]
    NdarrayError(#[from] ndarray::ShapeError),
}