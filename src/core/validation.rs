use crate::errors::DataError;
use ndarray::{ArrayBase, Dim};
use num_traits::Float;

pub fn validation_shape_2d_1d<T, S1, S2>(x: &ArrayBase<S1, Dim<[usize; 2]>>, y: &ArrayBase<S2, Dim<[usize; 1]>>) -> Result<(), DataError>
where
    T: Float,
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
{
    // --- 1. Verification of empty data ---
    if x.is_empty() {
        return Err(DataError::EmptyData);
    }

    if y.is_empty() {
        return Err(DataError::EmptyTarget);
    }

    // --- 2. Verification of NaN (Not a Number) ---
    if x.iter().any(|&e| e.is_nan()) {
        return Err(DataError::NaNData);
    }

    // --- 3. Verification of dimensional correspondence (Samples) ---
    let n_samples_x = x.shape()[0];
    let n_samples_y = y.shape()[0];

    if n_samples_x != n_samples_y {
        // Returns a DimensionMismatch error indicating the sizes found.
        return Err(DataError::DimensionMismatch(n_samples_x, n_samples_y));
    }

    Ok(())
}
