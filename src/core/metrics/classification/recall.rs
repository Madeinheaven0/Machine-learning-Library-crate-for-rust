use crate::core::validation::validation_shape_2d_1d;
use crate::errors::DataError;
use ndarray::Array1;
use std::hash::Hash;

// Necessary trait for class types (C)
pub trait RecallClass: Eq + Hash + Clone {}
impl RecallClass for usize {}
impl RecallClass for i32 {}
impl RecallClass for String {}

// ... implement for other class types if necessary
// Trait for the metric result (T: Float)
pub trait RecallFloat: num_traits::Float + num_traits::FromPrimitive {}
impl RecallFloat for f32 {}
impl RecallFloat for f64 {}

/// Calculates the Recall (Sensitivity) for a given positive class.
///
/// Args:
/// * y_true: True labels (Array1<C>).
/// * y_pred: Model predictions (Array1<C>).
/// * positive_class: The class considered positive (e.g., 1, "cat").
///
/// Returns: The recall rate (f64 or f32) or a DataError.
pub fn recall_binary<C, T>(
    y_true: &Array1<C>,
    y_pred: &Array1<C>,
    positive_class: &C,
) -> Result<T, DataError>
where
    C: RecallClass,
    T: RecallFloat,
{
    // Checking the size of the tables
    validation_shape_2d_1d(
        &y_true
            .view()
            .into_shape_with_order((y_true.len(), 1))
            .unwrap()
            .to_owned()
            .remove_axis(ndarray::Axis(1))
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap(),
        &y_pred
            .view()
            .into_shape_with_order((y_pred.len(), 1))
            .unwrap()
            .to_owned()
            .remove_axis(ndarray::Axis(1))
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap(),
    )?;

    let mut true_positives = 0; // VP
    let mut false_negatives = 0; // FN

    // Iterate over both arrays simultaneously
    for (y_t, y_p) in y_true.iter().zip(y_pred.iter()) {
        let is_actual_positive = y_t == positive_class;
        let is_predicted_positive = y_p == positive_class;

        if is_actual_positive {
            if is_predicted_positive {
               // The case was actually positive AND was predicted to be positive.
                true_positives += 1; // VP
            } else {
                // The case was actually positive BUT was predicted to be negative
                false_negatives += 1; // FN
            }
        }
    }

    // The denominator (VP + FN) is the total number of actual positive cases
    let denominator = true_positives + false_negatives;

    if denominator == 0 {
        // Cases where there are no actual positive cases (the recall is not applicable)
        return Err(DataError::InvalidData);
    }

    // Conversion to floating-point type T
    let vp_t = T::from_usize(true_positives).ok_or(DataError::InvalidData)?;
    let denom_t = T::from_usize(denominator).ok_or(DataError::InvalidData)?;

    Ok(vp_t / denom_t)
}

/// Calculates the Macro Callback (average of callbacks per class).
///
/// Args:
/// * y_true: True labels (Array1<C>).
/// * y_pred: Model predictions (Array1<C>).
///
/// Returns: The average Macro Callback (f64 or f32).
pub fn recall_macro<C, T>(y_true: &Array1<C>, y_pred: &Array1<C>) -> Result<T, DataError>
where
    C: RecallClass,
    T: RecallFloat,
{
    // 1. Identify all unique classes
    let unique_classes: Vec<C> = y_true
        .iter()
        .chain(y_pred.iter())
        .cloned()
        .collect::<std::collections::HashSet<C>>()
        .into_iter()
        .collect();

    let n_classes = unique_classes.len();
    if n_classes == 0 {
        return Ok(T::zero());
    }

    let mut recalls_sum = T::zero();

    // 2. Calculate the Recall for each class (using the binary function)
    for class in unique_classes.iter() {
        let recall = recall_binary(y_true, y_pred, class)?;
        recalls_sum = recalls_sum + recall;
    }

    // 3. Calculate the average
    let n_classes_t = T::from_usize(n_classes).ok_or(DataError::InvalidData)?;

    Ok(recalls_sum / n_classes_t)
}
