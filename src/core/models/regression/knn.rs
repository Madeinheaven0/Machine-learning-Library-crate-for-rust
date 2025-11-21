//! # KNN REGRESSOR

//! # KNN REGRESSOR

use crate::core::metrics::regression::linear::*;
use crate::core::metrics::regression::metrics::{Distance, EvaluationMetrics};
use crate::core::validation::validation_shape_2d_1d; // Importation de validation_shape_2d_1d
use crate::errors::DataError;
use ndarray::{Array1, Array2, ArrayBase};
use num_traits::{Float, FromPrimitive}; // Ajout de FromPrimitive
use std::fmt::Debug;

// Traits de base nécessaires pour les opérations (Float simplifie beaucoup)
pub trait KNNFloat: Float + FromPrimitive + Debug + Clone {}
// Implémentation du trait pour f32 et f64 (rend la contrainte plus propre)
impl KNNFloat for f32 {}
impl KNNFloat for f64 {}

pub struct KNNRegression<T>
where
    T: KNNFloat, // Utilisation du nouveau trait composite
{
    pub metrics: Distance,
    pub x_train: Option<Array2<T>>,
    pub y_train: Option<Array1<T>>,
    pub k: Option<usize>,
}

// Implementation
impl<T> KNNRegression<T>
where
    T: KNNFloat, // Contrainte simplifiée
{
    pub fn new(metrics: Distance) -> Self {
        Self {
            metrics,
            x_train: None,
            y_train: None,
            k: None,
        }
    }

    /// Stocke les features et la cible dans la structure après validation
    pub fn fit(&mut self, x: Array2<T>, y: Array1<T>, k: usize) -> Result<(), DataError> {
        // CORRECTION: validation_mix est pour un Array2<T> et un Array1<T> (OK ici)
        validation_shape_2d_1d(ArrayBase::from(&x), &y)?; // Vérifie uniquement les formes

        // Data storage
        self.x_train = Some(x);
        self.y_train = Some(y);
        self.k = Some(k);

        Ok(())
    }

    /// CORRECTION MAJEURE: Cette fonction semble faire de l'auto-prédiction sur le jeu d'entraînement.
    /// Elle est remplacée ou renommée pour clarifier son objectif.
    /// Le nom le plus commun est `predict_on_train` ou la supprimer si elle n'est pas nécessaire.
    /// Je l'ai supprimée, car la fonction `predict` gère déjà la prédiction sur de nouvelles données.
    /// Si vous voulez prédire sur X_train, appelez `self.predict(x_train)`

    /// Prédit les valeurs cibles pour un jeu de données de test (x_test).
    pub fn predict(&self, x_test: &Array2<T>) -> Result<Array1<T>, DataError> {
        // Vérification si le modèle est entraîné
        let x_train = self.x_train.as_ref().ok_or(DataError::NoteFittedModel)?;
        let y_train = self.y_train.as_ref().ok_or(DataError::NoteFittedModel)?;
        let k_val = self.k.ok_or(DataError::NoteFittedModel)?;
        let n_train = x_train.shape()[0];

        // x_test.shape()[0] est le nombre d'échantillons de test
        let mut predictions = Array1::zeros(x_test.shape()[0]);

        for i in 0..x_test.shape()[0] {
            let test_sample = x_test.row(i);

            // --- 1. Calculer la distance par rapport à x_train (DONNÉES D'ENTRAÎNEMENT) ---
            let distances_array: Array1<T> = match self.metrics {
                // L'opérateur de fermeture retourne Array1<T> ou une erreur.
                Distance::Euclidean => x_train
                    .outer_iter()
                    .map(|train_sample| {
                        let diff = &test_sample - &train_sample;
                        // On évite la racine carrée (sqrt) pour la performance car l'ordre est conservé
                        diff.map(|x| x.powi(2)).sum()
                    })
                    .collect(),

                Distance::Manhattan => x_train
                    .outer_iter()
                    .map(|train_sample| {
                        let diff = &test_sample - &train_sample;
                        diff.map(|x| x.abs()).sum()
                    })
                    .collect(),

                // CORRECTION MAJEURE E0308 : On retourne une erreur du scope de la fonction
                _ => return Err(DataError::InvalidMetric),
            };

            // ... (tri des voisins) ...
            let mut indexed_distances: Vec<(T, usize)> = distances_array
                .into_iter()
                .enumerate()
                .map(|(index, dist)| (dist.clone(), index))
                .collect();

            // sort by distance
            // La contrainte PartialOrd est vérifiée dans le .unwrap()
            indexed_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let limit = k_val.min(n_train);
            let k_nearest_indices: Vec<usize> = indexed_distances
                .into_iter()
                .take(limit)
                .map(|(_, index)| index)
                .collect();

            // --- 2. Calculer la moyenne des K voisins (Régression) ---
            let mut sum = T::zero();
            for &index in k_nearest_indices.iter() {
                // y_train contient les valeurs réelles de type T
                sum = sum + y_train[[index]].clone();
            }

            // CORRECTION E0308 : T::from(limit) attend un usize, pas un &usize.
            // On utilise .ok_or() car la conversion peut échouer.
            let mean = sum / T::from(limit).ok_or(DataError::InvalidData)?;
            predictions[[i]] = mean;
        }

        Ok(predictions)
    }

    /// Évalue les performances du modèle sur un jeu de test.
    pub fn evaluate(
        &self,
        x_test: &Array2<T>,
        y_test: &Array1<T>,
    ) -> Result<EvaluationMetrics<T>, DataError> {
        validation_shape_2d_1d(ArrayBase::from(x_test), y_test)?;

        let y_pred = self.predict(x_test)?;
        Ok(EvaluationMetrics {
            // CORRECTION E0308 : Ajout de `?` car les fonctions métriques retournent Result<T, DataError>
            mse: mse(y_test, &y_pred)?,
            r_squared: r_squared(y_test, &y_pred)?, // Remplacé rmse par r_squared pour la sémantique
            mae: mean_absolute_error(y_test, &y_pred)?,
        })
    }
}
