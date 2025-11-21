use crate::core::metrics::classification::recall::{RecallClass, RecallFloat};
use crate::errors::DataError;
use ndarray::Array1;
use std::collections::HashSet;
use std::hash::Hash;

/// Compute the  Precision for a given positive class.
///
/// Args:
/// * y_true: True labels (Array1<C>).
/// * y_pred: Les predicted labels (Array1<C>).
/// * positive_class: The chosen positive class (ex: 1, "cat").
///
/// Returns: The precision rate (f64 or f32) or a DataError.
pub fn precision_binary<C, T>(
    y_true: &Array1<C>,
    y_pred: &Array1<C>,
    positive_class: &C,
) -> Result<T, DataError>
where
    C: RecallClass,
    T: RecallFloat,
{
    // Validating array size (using a temporary view of Array1<C> as if it were Array1<T>)
    // NOTE: C-specific type validation would be required, or ensuring that C implements validation traits.
    // Assume that y_true.len() == y_pred.len() is sufficient here.
    if y_true.len() != y_pred.len() {
        return Err(DataError::DimensionMismatch(y_true.len(), y_pred.len()));
    }

    let mut true_positives = 0; // VP
    let mut false_positives = 0; // FP

    // Iterate over both arrays simultaneously
    for (y_t, y_p) in y_true.iter().zip(y_pred.iter()) {
        let is_predicted_positive = y_p == positive_class;

        if is_predicted_positive {
            if y_t == positive_class {
                // The case was predicted to be positive AND is actually positive.
                true_positives += 1; // VP
            } else {
                // The case was predicted to be positive BUT is actually negative.
                false_positives += 1; // FP
            }
        }
    }

    // The denominator (VP + FP) is the total number of positive predictions
    let denominator = true_positives + false_positives;

    if denominator == 0 {
        // Case where no positive prediction was made.
        // Precision is usually undefined (or 1.0 by convention if VP=0).
        // We return an error or 1.0 (according to convention). Let's choose T::zero() here if undefined, or InvalidData.
        return Ok(T::zero());
    }

    // Conversion to floating-point type T
    let vp_t = T::from_usize(true_positives).ok_or(DataError::InvalidData)?;
    let denom_t = T::from_usize(denominator).ok_or(DataError::InvalidData)?;

    Ok(vp_t / denom_t)
}

/// Calculates the Macro Precision (average of precisions per class).
///
/// Args:
/// * y_true: True labels (Array1<C>).
/// * y_pred: The model predictions (Array1<C>).
///
/// Returns: Average macro precision (f64 or f32).
pub fn precision_macro<C, T>(y_true: &Array1<C>, y_pred: &Array1<C>) -> Result<T, DataError>
where
    C: RecallClass,
    T: RecallFloat,
{
    // 1. Identify all unique classes
    let unique_classes: HashSet<C> = y_true.iter().chain(y_pred.iter()).cloned().collect();

    let n_classes = unique_classes.len();
    if n_classes == 0 {
        return Ok(T::zero());
    }

    let mut precisions_sum = T::zero();
    let mut valid_classes_count = 0;

    // 2. Calculate the precision for each class (using the binary function)
    for class in unique_classes.iter() {
        // The result can be T::zero() if the denominator is 0.
        // We treat this case as a valid class for the macro average.
        let precision = precision_binary(y_true, y_pred, class)?;

        precisions_sum = precisions_sum + precision;
        valid_classes_count += 1;
    }

    // 3. Calculate the average
    if valid_classes_count == 0 {
        return Ok(T::zero()); // No class with which to calculate the average;
    }

    let n_classes_t = T::from_usize(valid_classes_count).ok_or(DataError::InvalidData)?;

    Ok(precisions_sum / n_classes_t)
}
