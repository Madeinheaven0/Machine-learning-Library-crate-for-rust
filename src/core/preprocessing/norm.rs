use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use super::class::PreprocessorClass;
use ndarray::{Dimension, Array, Axis};
use ndarray_stats::{QuantileExt};
use crate::errors::DataError;

fn standard_scaler<A, D>(arr: Array<A, D>) -> Result<Array<A, D>, DataError>
where
    A: Add + Sub +
    Mul + Div +
    Rem + Neg +
    PartialOrd + PartialEq +
    Clone + Debug ,
    D: Dimension, {
    if arr.is_empty() {
        return Err(DataError::EmptyData);
    }

    let dim_count = arr.ndim();

    match dim_count {
        1 => {
            let mean = arr.mean().expect("We can compute the mean");
            let std = arr.std(1.);

            Ok(
                (arr - mean) / std
            )
        },
        2 => {
            let mean = arr.mean_axis(Axis(0)).expect("We can compute the mean");
            let std = arr.std_axis(Axis(0), 1.);

            Ok(
                (arr - mean) / std
            )
        },

        _ => Err(DataError::DimensionMismatch(arr.shape()[0], arr.shape()[1]))
    }
}

fn minmax_scaler<A, D>(arr: Array<A, D>) -> Result<Array<A, D>, DataError>
where
    A: Add + Sub +
    Mul + Div +
    Rem + Neg +
    PartialOrd + PartialEq +
    Clone + Debug,
    D: Dimension, {
    if arr.is_empty() {
        return Err(DataError::EmptyData);
    }

    let dim_count = arr.ndim();

    match dim_count {
        1 => {
            let min = arr.min()?.clone();
            let max = arr.max()?.clone();

            Ok(
                (arr - &min) / (max - min)
            )
        },
        2 => {
            let min = arr.map_axis(Axis(0), |c| c.min()).iter().collect::<Array<A, D>>();
            let max = arr.map_axis(Axis(1), |c| c.max()).iter().collect::<Array<A, D>>();

            Ok(
                (arr - &min) / (max - min)
            )
        },
        _ => Err(DataError::DimensionMismatch(arr.shape()[0], arr.shape()[1]))
    }
}

fn robust_scaler<A, D>(arr: Array<A, D>) -> Result<Array<A, D>, DataError>
where
    A: Add + Sub +
    Mul + Div +
    Rem + Neg +
    PartialOrd + PartialEq +
    Clone + Debug,
    D: Dimension, {
    if arr.is_empty() {
        return Err(DataError::EmptyData);
    }

    let dim_count = arr.ndim();

    todo!()
}

fn preprocessor<A, D>(arr: Array<A, D>, cat: PreprocessorClass) -> Result<Array<A, D>, DataError>
where
    A: Add + Sub +
    Mul + Div +
    Rem + Neg +
    PartialOrd + PartialEq +
    Clone + Debug,
    D: Dimension, {
    match cat {
        PreprocessorClass::MinMax => minmax_scaler(arr),
        PreprocessorClass::Standard => standard_scaler(arr),
        PreprocessorClass::Robust => robust_scaler(arr),
    }
}