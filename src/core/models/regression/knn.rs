//! # KNN REGRESSOR

use crate::core::metrics::regression::linear::*;
use crate::core::metrics::regression::metrics::{Distance, EvaluationMetrics};
use crate::core::validation::*;
use crate::errors::DataError;
use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

pub struct KNNRegression<T>
where
    T: Add
        + Sub
        + Mul
        + Div
        + Rem
        + Neg
        + Sub
        + Add
        + PartialOrd
        + PartialEq
        + Clone
        + Debug
        + Float,
{
    pub metrics: Distance,
    pub x_train: Option<Array2<T>>,
    pub y_train: Option<Array1<T>>,
    pub k: Option<usize>,
}

// Implementation
impl<T> KNNRegression<T>
where
    T: Add
        + Sub
        + Mul
        + Div
        + Rem
        + Neg
        + Sub
        + Add
        + PartialOrd
        + PartialEq
        + Ord
        + Eq
        + Clone
        + Debug
        + Float
        + Debug,
{
    pub fn new(metrics: Distance) -> Self {
        Self {
            metrics,
            x_train: None,
            y_train: None,
            k: None,
        }
    }

    // Storage the feature and the target in the structure
    pub fn fit(&mut self, x: Array2<T>, y: Array1<T>, k: usize) -> Result<(), DataError> {
        validation_mix(&x, &y)?;

        // Data storage
        self.x_train = Some(x);
        self.y_train = Some(y);
        self.k = Some(k);

        Ok(())
    }

    pub fn predicted(&self) -> Result<Array1<T>, DataError> {
        // Verify if the model is trained
        let x_train = self.x_train.as_ref().ok_or(DataError::NoteFittedModel)?;
        let y_train = self.y_train.as_ref().ok_or(DataError::NoteFittedModel)?;
        let k_val = self.k.ok_or(DataError::NoteFittedModel)?;

        let mut predictions = Array1::zeros(y_train.shape()[0]);

        //--- 1. Compute the distances on the base of the metrics
        for i in 0..y_train.shape()[0] {
            let test_sample = x_train.row(i);
            let n_train = x_train.shape()[0];

            //--- 1. Compute the distance on the base of the metrics
            let distances_array: Array1<T> = match self.metrics {
                // Euclidean distance (L2) : sqrt(sum((a_1 - b_i) ^ 2))
                Distance::Euclidean => x_train
                    .outer_iter()
                    .map(|train_sample| {
                        let diff = &test_sample - &train_sample;

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

                _ => Err(DataError::InvalidMetric),
            };

            let mut indexed_distances: Vec<(T, usize)> = distances_array
                .into_iter()
                .enumerate()
                .map(|(index, dist)| (dist.clone(), index))
                .collect();

            // sort by distance
            indexed_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let limit = k_val.min(n_train);
            let k_nearest_indices: Vec<usize> = indexed_distances
                .into_iter()
                .take(limit)
                .map(|(_, index)| index)
                .collect();

            let mut sum = T::zero();
            for &index in k_nearest_indices.iter() {
                sum = sum + y_train[[index]].clone();
            }

            let mean = sum / T::from(&limit).unwrap();
            predictions[[i]] = mean;
        }

        Ok(predictions)
    }

    // La signature devrait accepter Array2<T> pour les features de test
    pub fn predict(&self, x_test: Array2<T>) -> Result<Array1<T>, DataError> {
        // --- NOUVEAU: Récupérer les données d'entraînement ---
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
                Distance::Euclidean => x_train // <--- CORRECTION
                    .outer_iter()
                    .map(|train_sample| {
                        let diff = &test_sample - &train_sample;
                        diff.map(|x| x.powi(2)).sum()
                    })
                    .collect(),

                Distance::Manhattan => x_train // <--- CORRECTION
                    .outer_iter()
                    .map(|train_sample| {
                        let diff = &test_sample - &train_sample;
                        diff.map(|x| x.abs()).sum()
                    })
                    .collect(),

                _ => return Err(DataError::InvalidMetric),
            };

            // ... (tri des voisins) ...
            let mut indexed_distances: Vec<(T, usize)> = distances_array
                .into_iter()
                .enumerate()
                .map(|(index, dist)| (dist.clone(), index))
                .collect();

            // sort by distance
            indexed_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let limit = k_val.min(n_train);
            let k_nearest_indices: Vec<usize> = indexed_distances
                .into_iter()
                .take(limit)
                .map(|(_, index)| index)
                .collect();

            let mut sum = T::zero();
            for &index in k_nearest_indices.iter() {
                sum = sum + y_train[[index]].clone();
            }

            let mean = sum / T::from(limit).unwrap();
            predictions[[i]] = mean;
        }

        Ok(predictions)
    }

    pub fn evaluate(
        &self,
        y_test: &Array2<T>,
        y_pred: &Array1<T>,
    ) -> Result<EvaluationMetrics<T>, DataError> {
        validation_mix(y_test, y_pred)?;

        Ok(EvaluationMetrics {
            mse: mse(&y_test, &y_pred),
            r_squared: rmse(&y_test, &y_pred),
            mae: mean_absolute_error(&y_test, &y_pred),
        })
    }
}
