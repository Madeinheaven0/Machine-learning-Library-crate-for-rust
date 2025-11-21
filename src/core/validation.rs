use crate::errors::DataError;
use ndarray::{ArrayBase, Dim};
use num_traits::Float;

pub fn validation_shape_2d_1d<T, S1, S2>(x: &ArrayBase<S1, Dim<[usize; 2]>>, y: &ArrayBase<S2, Dim<[usize; 1]>>) -> Result<(), DataError>
where
    T: Float,
    S1: ndarray::Data<Elem = T>, // T doit être le type des éléments
    S2: ndarray::Data<Elem = T>,
{
    // --- 1. Vérification des données vides ---
    if x.is_empty() {
        return Err(DataError::EmptyData);
    }

    if y.is_empty() {
        return Err(DataError::EmptyTarget);
    }

    // --- 2. Vérification des NaN (Not a Number) ---
    // Utilise le trait Float pour appeler .is_nan() sur chaque élément de X.
    if x.iter().any(|&e| e.is_nan()) {
        return Err(DataError::NaNData);
    }

    // --- 3. Vérification de la correspondance des dimensions (Échantillons) ---
    let n_samples_x = x.shape()[0]; // Nombre de lignes dans X
    let n_samples_y = y.shape()[0]; // Nombre d'éléments dans Y

    if n_samples_x != n_samples_y {
        // Retourne une erreur DimensionMismatch indiquant les tailles trouvées.
        return Err(DataError::DimensionMismatch(n_samples_x, n_samples_y));
    }

    Ok(())
}
