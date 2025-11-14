use crate::errors::DataError;
use ndarray::{Array1, Array2};
use num_traits::Float;

pub fn validation_mix<T: Float + PartialOrd + PartialEq>(
    x: &Array2<T>,
    y: &Array1<T>,
) -> Result<(), DataError> {
    if x.is_empty() || y.is_empty() {
        return Err(DataError::EmptyData);
    }

    if x.shape()[0] != y.shape()[0] {
        return Err(DataError::DimensionMismatch(x.shape()[0], y.shape()[0]));
    }

    if x.iter().any(|&x| x.is_nan()) || y.iter().any(|&x| x.is_nan()) {
        return Err(DataError::NaNData);
    }

    Ok(())
}

pub fn validation_features<T: Float + PartialOrd + PartialEq>(
    x: &Array2<T>,
) -> Result<(), DataError> {
    if x.is_empty() {
        return Err(DataError::EmptyData);
    }

    if x.iter().any(|&x| x.is_nan()) {
        return Err(DataError::NaNData);
    }

    Ok(())
}

pub fn validation_target<T: Float + PartialOrd + PartialEq>(
    y: &Array1<T>,
) -> Result<(), DataError> {
    if y.is_empty() {
        return Err(DataError::EmptyTarget);
    }

    if y.iter().any(|&x| x.is_nan()) {
        return Err(DataError::NaNData);
    }

    Ok(())
}
