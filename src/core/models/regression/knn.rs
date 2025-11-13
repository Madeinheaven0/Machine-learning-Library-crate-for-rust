use crate::core::metrics::regression::metrics::Distance;
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
        }
    }

    pub fn fit(&mut self, x: Array2<T>, y: Array1<T>) -> Result<(), DataError> {
        if x.is_empty() || y.is_empty() {
            return Err(DataError::EmptyData);
        }

        if x.shape()[0] != y.shape()[0] {
            return Err(DataError::DimensionMismatch(x.shape()[0], y.shape()[0]));
        }

        // Nan verification
        if x.iter().any(|val| val.is_nan()) || y.iter().any(|val| val.is_nan()) {
            return Err(DataError::NaNData);
        }

        // Data storage
        self.x_train = Some(x);
        self.y_train = Some(y);

        Ok(())
    }

    pub fn predict(&self, x_test: &Array2<T>, k: usize) -> Result<Array1<T>, DataError> {
        // Verify if the model is trained
        let x_train = self.x_train.as_ref().ok_or(DataError::NoteFittedModel)?;
        let y_train = self.y_train.as_ref().ok_or(DataError::NoteFittedModel)?;

        let mut predictions = Array1::zeros(y_train.shape()[0]);

        //--- 1. Compute the distances on the base of the metrics
        for i in 0..n_test {
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

            let mut indexed_distances: Vec<(T, usize)> = distances_array.into_iter()
                .enumerate()
                .map(|(index, dist)| (dist.clone(), index))
                .collect();

            // sort by distance
            indexed_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let limit  = k.min(n_train);
            let k_nearest_indices: Vec<usize> = indexed_distances.into_iter()
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
}
