use crate::core::metrics::classification::precision::precision_binary;
use crate::core::metrics::classification::recall::{RecallClass, RecallFloat, recall_binary};
use crate::errors::DataError;
use ndarray::Array1;
use std::collections::HashSet;

/// ## 1. Binary f1 score

/// Compute the f1 score for a given positive class (harmonic average of the Precision and Recall).
///
/// Args:
/// * y_true: True labels (Array1<C>).
/// * y_pred: Predicted labels (Array1<C>).
/// * positive_class: The positive chosen class.
///
/// Returns: The f1 score(f64 or f32) or a DataError.
pub fn f1_score_binary<C, T>(
    y_true: &Array1<C>,
    y_pred: &Array1<C>,
    positive_class: &C,
) -> Result<T, DataError>
where
    C: RecallClass,
    T: RecallFloat,
{
    // 1. Compute the precision and the recall
    let precision = precision_binary(y_true, y_pred, positive_class)?;
    let recall = recall_binary(y_true, y_pred, positive_class)?;

    // 2. Compute the denominator of the harmonic mean (Precision + Recall)
    let denominator = precision + recall;

    // If the denominator is 0 (case where Precision and Rappel are both  equal to 0),
    // The F1 Score is 0.
    if denominator.is_zero() {
        return Ok(T::zero());
    }

    // 3. Apply the F1 formula = 2 * (Precision * Recall) / (Precision + Recall)
    let numerator = precision * recall;
    let two = T::from_f64(2.0).ok_or(DataError::InvalidData)?; // Conversion of 2.0 in type T

    Ok(two * numerator / denominator)
}

/// ## 2. Score F1 Macro mean

/// Compute the f1 score Macro (mean of f1 Scores per class).
///
/// Args:
/// * y_true: True labels (Array1<C>).
/// * y_pred: Predicted labels (Array1<C>).
///
/// Returns: F1 Score Macro mean (f64 or f32).
pub fn f1_score_macro<C, T>(y_true: &Array1<C>, y_pred: &Array1<C>) -> Result<T, DataError>
where
    C: RecallClass,
    T: RecallFloat,
{
    // 1. Identify all the unique classes
    let unique_classes: HashSet<C> = y_true.iter().chain(y_pred.iter()).cloned().collect();

    let n_classes = unique_classes.len();
    if n_classes == 0 {
        return Ok(T::zero());
    }

    let mut f1_scores_sum = T::zero();

    // 2. Compute the F1 Score F1 for each class
    for class in unique_classes.iter() {
        let f1 = f1_score_binary(y_true, y_pred, class)?;
        f1_scores_sum = f1_scores_sum + f1;
    }

    // 3. Compute the mean
    let n_classes_t = T::from_usize(n_classes).ok_or(DataError::InvalidData)?;

    Ok(f1_scores_sum / n_classes_t)
}
