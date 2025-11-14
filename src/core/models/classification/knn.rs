use crate::core::metrics::regression::metrics::Distance;
use crate::core::validation::*;
use crate::errors::DataError;
use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

pub struct KNNClassifier<T, C>
where
    T: Float + Clone + Debug + PartialOrd + PartialEq, // T pour les features (distances)
    C: Eq + Hash + Clone + Debug,                      // C pour les classes
{
    pub metrics: Distance,
    pub x_train: Option<Array2<T>>,
    pub y_train: Option<Array1<C>>,
    pub k: Option<usize>,
}

impl<T, C> KNNClassifier<T, C>
where
    T: Float + Add + Sub + Mul + Div + Rem + Neg + PartialOrd + PartialEq + Clone + Debug,
    C: Eq + Hash + Clone + Debug,
{
    pub fn new(metrics: Distance) -> Self {
        Self {
            metrics,
            x_train: None,
            y_train: None,
            k: None,
        }
    }

    // Stores the features (Array2<T>) and the targets (Array1<C>)
    pub fn fit(&mut self, x: Array2<T>, y: Array1<C>, k: usize) -> Result<(), DataError> {
        validation_mix(&x, &y)?;

        // Data storage (Mémorisation des données)
        self.x_train = Some(x);
        self.y_train = Some(y);
        self.k = Some(k);

        Ok(())
    }

    // Makes predictions for a new dataset (x_test)
    pub fn predict(&self, x_test: Array2<T>) -> Result<Array1<C>, DataError> {
        // 1. Verification of the trained model
        let x_train = self.x_train.as_ref().ok_or(DataError::NoteFittedModel)?;
        let y_train = self.y_train.as_ref().ok_or(DataError::NoteFittedModel)?;
        let k_val = self.k.ok_or(DataError::NoteFittedModel)?;
        let n_train = x_train.shape()[0];

        if x_test.shape()[1] != x_train.shape()[1] {
            return Err(DataError::DimensionMismatch(
                x_test.shape()[1],
                x_train.shape()[1],
            ));
        }

        // Initializing the prediction array (Array1<C>)

        // Requires cloning an existing class to initialize the array
        let default_class = y_train.get([0]).ok_or(DataError::NoteFittedModel)?.clone();
        let mut predictions = Array1::from_elem(x_test.shape()[0], default_class);

        // Loop through each test sample
        for i in 0..x_test.shape()[0] {
            let test_sample = x_test.row(i);

            // --- 2. Calculating Distances (Same as Regression) ---

            let distances_array: Array1<T> = match self.metrics {
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

                _ => return Err(DataError::InvalidMetric),
            };

            // --- 3. Sorting and Selecting Neighbors ---

            // Associates each distance with its index (distance, index)
            let mut indexed_distances: Vec<(T, usize)> = distances_array
                .into_iter()
                .enumerate()
                .map(|(index, dist)| (dist.clone(), index))
                .collect();

            // Sort by increasing distance (smallest first)
            indexed_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let limit = k_val.min(n_train);

            // Isolates the indices of the K nearest neighbors
            let k_nearest_indices: Vec<usize> = indexed_distances
                .into_iter()
                .take(limit)
                .map(|(_, index)| index)
                .collect();

            // --- 4. Classification Vote (Key Difference) ---

            let mut class_counts: HashMap<C, usize> = HashMap::new();

            // Count the votes of the K neighbors in y_train
            for &index in k_nearest_indices.iter() {
                let class = y_train[[index]].clone();
                // Increments the counter for this class
                *class_counts.entry(class).or_insert(0) += 1;
            }

            // Determine the majority class
            // .max_by_key finds the class with the largest 'count'
            let (predicted_class, _) = class_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .ok_or(DataError::DimensionMismatch(1, 1))?;

            // 5. Store the prediction
            predictions[[i]] = predicted_class.clone();
        }

        Ok(predictions)
    }
}
