use ndarray::Array1;
// [Ajustement des imports et des types]
// ...
use crate::core::validation::validation_shape_2d_1d; // Assumer ce nom de fonction de validation
use crate::errors::DataError;
use num_traits::{Float, FromPrimitive};
// Import FromPrimitive pour T::from()
// ...

// Helper pour la conversion safe (déjà bien fait)
fn usize_to_t<T: Float + FromPrimitive>(n: usize) -> Result<T, DataError> {
    T::from(n).ok_or(DataError::InvalidData) // Changé NaNData pour InvalidData, car l'échec de conversion n'est pas NaN
}

// --- Fonctions de Métriques ---

pub fn mse<T: Float + FromPrimitive>(x: &Array1<T>, y: &Array1<T>) -> Result<T, DataError> {
    // CORRECTION: Utiliser la fonction de validation créée précédemment
    validation_shape_2d_1d(x.view().into_shape_with_order((x.len(), 1)).unwrap(), y)?;

    let n_samples = usize_to_t(x.shape()[0])?;

    // La différence (x - y) est le vecteur d'erreur
    let error = x - y;

    // Erreur au carré (différence * différence)
    let squared_error: T = error.mapv(|e| e * e).sum();

    Ok(squared_error / n_samples)
}

pub fn rmse<T: Float + FromPrimitive>(x: &Array1<T>, y: &Array1<T>) -> Result<T, DataError> {
    // CORRECTION: Appeler mse et prendre sa racine carrée
    let mean_sq_err = mse(x, y)?;
    Ok(mean_sq_err.sqrt())
}

// Nouvelle fonction essentielle : R-Squared (Coefficient de Détermination)
pub fn r_squared<T: Float + FromPrimitive>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
) -> Result<T, DataError> {
    // Vrai y d'abord, prédiction ensuite (convention standard)
    validation_shape_2d_1d(
        y_true
            .view()
            .into_shape_with_order((y_true.len(), 1))
            .unwrap(),
        y_pred,
    )?;

    // 1. Calculer la moyenne de y_true
    let n_samples = usize_to_t(y_true.shape()[0])?;
    let y_mean = y_true.sum() / n_samples;

    // 2. Erreur résiduelle (Sum of Squares Residual - SSR) : (y_true - y_pred)^2
    let ssr: T = (y_true - y_pred).mapv(|e| e * e).sum();

    // 3. Erreur totale (Sum of Squares Total - SST) : (y_true - y_mean)^2
    let sst: T = y_true.mapv(|y| y - y_mean).mapv(|e| e * e).sum();

    // R^2 = 1 - (SSR / SST)
    // Protection contre la division par zéro (SST = 0 si toutes les vraies valeurs sont identiques)
    if sst.is_zero() {
        // Si SST est zéro, R² est indéfini ou peut être considéré comme 1.0 si SSR est 0
        return if ssr.is_zero() {
            Ok(T::one())
        } else {
            Err(DataError::InvalidData)
        };
    }

    Ok(T::one() - (ssr / sst))
}

pub fn lmse<T: Float + FromPrimitive>(x: &Array1<T>, y: &Array1<T>) -> Result<T, DataError> {
    // Appelle MSE et prend le log naturel
    let mean_sq_err = mse(x, y)?;
    Ok(mean_sq_err.ln())
}

pub fn lrmse<T: Float + FromPrimitive>(x: &Array1<T>, y: &Array1<T>) -> Result<T, DataError> {
    // Appelle RMSE et prend le log naturel
    let root_mean_sq_err = rmse(x, y)?;
    Ok(root_mean_sq_err.ln())
}

pub fn mean_absolute_error<T: Float + FromPrimitive>(
    x: &Array1<T>,
    y: &Array1<T>,
) -> Result<T, DataError> {
    // Utiliser la validation standard pour gérer les NaNs et la forme
    validation_shape_2d_1d(x.view().into_shape_with_order((x.len(), 1)).unwrap(), y)?;

    let n_samples = usize_to_t(x.shape()[0])?;

    // Erreur absolue (|x - y|)
    let absolute_error: T = (x - y).mapv(|e| e.abs()).sum();

    Ok(absolute_error / n_samples)
}

pub fn regression_metric<T: Float + FromPrimitive>(
    x: Array1<T>, // Devrait être y_true
    y: Array1<T>, // Devrait être y_pred
    metric: RegressionMetric,
) -> Result<T, DataError> {
    // CORRECTION: Ajouter le R-Squared et vérifier l'appel des fonctions
    let measure = match metric {
        RegressionMetric::MeanSquareError => mse(&x, &y)?,
        RegressionMetric::MeanAbsoluteError => mean_absolute_error(&x, &y)?,
        RegressionMetric::RootMeanSquaredError => rmse(&x, &y)?,
        // Ajout du Coefficient de Détermination
        RegressionMetric::RSquared => r_squared(&x, &y)?,
        RegressionMetric::LogMeanSquaredError => lmse(&x, &y)?,
        RegressionMetric::LogRootMeanSquaredError => lrmse(&x, &y)?,
        // NOTE: Si vous aviez RegressionMetric::RootMeanSquaredError qui appelait rmse,
        // vous devriez renommer la variable d'entrée 'x' en 'y_true' pour la clarté.
    };

    Ok(measure)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionMetric {
    /// Erreur Quadratique Moyenne (Mean Squared Error).
    MeanSquareError,

    /// Erreur Absolue Moyenne (Mean Absolute Error).
    MeanAbsoluteError,

    /// Racine Carrée de l'Erreur Quadratique Moyenne (Root Mean Squared Error).
    RootMeanSquaredError,

    /// Coefficient de Détermination (R-Squared).
    RSquared,

    /// Logarithme de l'Erreur Quadratique Moyenne (Log Mean Squared Error).
    LogMeanSquaredError,

    /// Logarithme de la Racine Carrée de l'Erreur Quadratique Moyenne
    /// (Log Root Mean Squared Error).
    LogRootMeanSquaredError,
}
