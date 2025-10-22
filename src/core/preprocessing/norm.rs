use std::ops::{Div, Sub};
use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};
use ndarray_rand::rand_distr::num_traits::Float;
use crate::core::traits::{StatisticsOps, TensorLike};
use crate::errors::DataError;

// Alias for ArrayD<T::Elem> in the context of T
type ArrayDOutput<T> = ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>;

fn min_max_scaler<T: StatisticsOps>(data: &T) -> Result<ArrayD<T::Elem>, DataError>
where
    <T as TensorLike>::Elem: Float,
//1. Constraint for subtraction (Numerator)
    for<'a> &'a T: Sub<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>,

// 2. Constraint for division (Added by compiler).
    for<'a> <&'a T as Sub<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>>::Output:
    Div<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>,

// 3.THE FINAL CONSTRAINT: Force the output type to be ArrayD
    for<'a> <
    <&'a T as Sub<&'a ArrayDOutput<T>>>::Output as Div<&'a ArrayDOutput<T>>
    >::Output: Into<ArrayDOutput<T>>,
{
    let min_data = data.min()?;
    let max_data = data.max()?;

    // 1. Calculate the numerator: eT - Array = Array
    let numerator = data - &min_data;

    // 2. Calculate the denominator: Array - Array = Array
    let range = max_data - &min_data;

    // 3. Division : ArrayD / &ArrayD
    Ok((numerator / &range).into())
}

fn standard_scaler<T: StatisticsOps>(data: &T) -> Result<ArrayD<T::Elem>, DataError>
where
    <T as TensorLike>::Elem: Float,
//1. Constraint for subtraction (Numerator)
    for<'a> &'a T: Sub<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>,

// 2. Constraint for division (Added by compiler).
    for<'a> <&'a T as Sub<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>>::Output:
    Div<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>,

// 3.THE FINAL CONSTRAINT: Force the output type to be ArrayD
    for<'a> <
    <&'a T as Sub<&'a ArrayDOutput<T>>>::Output as Div<&'a ArrayDOutput<T>>
    >::Output: Into<ArrayDOutput<T>>,
{
    let mean_data = data.mean()?;
    let std_data = data.std()?;

    Ok(((data - &mean_data) / &std_data).into())
}


fn robust_scaler<T: StatisticsOps>(data: &T) -> Result<ArrayD<T::Elem>, DataError>
where
    <T as TensorLike>::Elem: Float,
    for<'a> &'a T: Sub<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>,
    for<'a> <&'a T as Sub<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>>::Output:
    Div<&'a ArrayBase<OwnedRepr<<T as TensorLike>::Elem>, Dim<IxDynImpl>>>,
    for<'a> <
    <&'a T as Sub<&'a ArrayDOutput<T>>>::Output as Div<&'a ArrayDOutput<T>>
    >::Output: Into<ArrayDOutput<T>>,
{
    let median_data = data.median()?;
    let first_quantile = data.first_quantile()?;
    let third_quantile = data.third_quantile()?;

    Ok(((data - &median_data) / &((&third_quantile - &first_quantile))).into())
}