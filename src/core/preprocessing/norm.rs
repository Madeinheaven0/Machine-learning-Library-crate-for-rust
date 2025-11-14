use super::class::PreprocessorClass;
use crate::errors::DataError;
use ndarray::{Array, Array1, Array2, Axis, Dimension};
use ndarray_stats::QuantileExt;
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

// This function calculates a quantile (Q1, Median, Q3) for an Array1<A>
// The 'q' is the position, e.g., 0.5 for the median, 0.25 for Q1.
fn quantile<A>(arr: &Array1<A>, q: f64) -> A
where
    A: Float + PartialOrd + Clone + Debug,
{
    let mut data = arr.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data.len();

    // Position of the index based on type A (Float)
    let index_f = A::from(n).unwrap() * A::from(q).unwrap();
    let index = index_f.to_usize().unwrap().max(1) - 1; // Adjustment for 0-based indexing

    // Simple method: returns the value at the nearest index
    data[index].clone()

    // NOTE: A more rigorous implementation would interpolate between
    // two points if the index is not an integer, but this approximation
    // is sufficient for standard scaling.
}

fn standard_scaler<A>(arr: &Array2<A>) -> Result<Array2<A>, DataError>
where
    A: Add + Sub + Mul + Div + Rem + Neg + PartialOrd + PartialEq + Clone + Debug + Float,
{
    if arr.is_empty() {
        return Err(DataError::EmptyData);
    }

    let dim_count = arr.ndim();

    match dim_count {
        1 => {
            let mean = arr.mean().expect("We can compute the mean");
            let std = arr.std(1.);

            Ok((arr - mean) / std)
        }
        2 => {
            let mean = arr.mean_axis(Axis(0)).expect("We can compute the mean");
            let std = arr.std_axis(Axis(0), 1.);

            Ok((arr - mean) / std)
        }

        _ => Err(DataError::DimensionMismatch(arr.shape()[0], arr.shape()[1])),
    }
}

fn minmax_scaler<A>(arr: &Array2<A>) -> Result<Array2<A>, DataError>
where
    A: Add + Sub + Mul + Div + Rem + Neg + PartialOrd + PartialEq + Clone + Debug,
{
    if arr.is_empty() {
        return Err(DataError::EmptyData);
    }

    let dim_count = arr.ndim();

    match dim_count {
        1 => {
            let min = arr.min()?.clone();
            let max = arr.max()?.clone();

            Ok((arr - &min) / (max - min))
        }
        2 => {
            let min = arr
                .map_axis(Axis(0), |c| c.min())
                .iter()
                .collect::<Array2<A>>();
            let max = arr
                .map_axis(Axis(1), |c| c.max())
                .iter()
                .collect::<Array2<A>>();

            Ok((arr - &min) / (max - min))
        }
        _ => Err(DataError::DimensionMismatch(arr.shape()[0], arr.shape()[1])),
    }
}

fn robust_scaler<A>(arr: &Array2<A>) -> Result<Array2<A>, DataError>
where
    A: Float + Add + Sub + Mul + Div + Rem + Neg + PartialOrd + PartialEq + Clone + Debug,
{
    if arr.is_empty() {
        return Err(DataError::EmptyData);
    }

    // We will implement the 2D case (columns as features)
    let dim_count = arr.ndim();

    match dim_count {
        // The 1D case (simple vector) is trivial
        1 => {
            let median = quantile(&arr.clone().into_dimensionality().unwrap(), 0.5);
            let q1 = quantile(&arr.clone().into_dimensionality().unwrap(), 0.25);
            let q3 = quantile(&arr.clone().into_dimensionality().unwrap(), 0.75);

            let iqr = q3 - q1;

            if iqr == A::zero() {
                return Err(DataError::CalculationError("IQR is zero".to_string()));
            }

            Ok((arr - median) / iqr)
        }

        // The 2D case is the most relevant
        2 => {
            // Calculate Q1, Median and Q3 for EACH COLUMN (Axis(0))
            let q1_array: Array1<A> = arr
                .columns() // Iterates over the columns
                .into_iter()
                .map(|col| quantile(&col.to_owned().into_dimensionality().unwrap(), 0.25))
                .collect();

            let median_array: Array1<A> = arr
                .columns()
                .into_iter()
                .map(|col| quantile(&col.to_owned().into_dimensionality().unwrap(), 0.5))
                .collect();

            let q3_array: Array1<A> = arr
                .columns()
                .into_iter()
                .map(|col| quantile(&col.to_owned().into_dimensionality().unwrap(), 0.75))
                .collect();

            // 1. Calculate the IQR (Q3 - Q1)
            let iqr = q3_array - q1_array;

            // 2. Check if the IQR is zero to avoid division by zero
            if iqr.iter().any(|&v| v == A::zero()) {
                return Err(DataError::CalculationError(
                    "One quantile is zero".to_string(),
                ));
            }

            // 3. Scaling
            // (arr - median) / iqr
            // NOTE: We must force median_array and iqr to be 2D views (1 line)
            // so that ndarray can perform the operation line by line.

            let median_2d = median_array.insert_axis(Axis(0));
            let iqr_2d = iqr.insert_axis(Axis(0));

            Ok((arr - median_2d) / iqr_2d)
        }

        // Manage larger dimensions
        _ => Err(DataError::DimensionMismatch(arr.shape()[0], arr.shape()[1])),
    }
}

fn preprocessor<A>(arr: Array2<A>, cat: PreprocessorClass) -> Result<Array2<A>, DataError>
where
    A: Add + Sub + Mul + Div + Rem + Neg + PartialOrd + PartialEq + Clone + Debug + Float,
{
    match cat {
        PreprocessorClass::MinMax => minmax_scaler(&arr),
        PreprocessorClass::Standard => standard_scaler(&arr),
        PreprocessorClass::Robust => robust_scaler(&arr),
    }
}
