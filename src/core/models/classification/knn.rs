use crate::core::metrics::regression::metrics::Distance;
use crate::errors::DataError;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

// Definition of the composite stroke for clarity
pub trait KNNFloat: Float + Clone + Debug {}
impl KNNFloat for f32 {}
impl KNNFloat for f64 {}

// Definition of the structure with simplified constraints
pub struct KNNClassifier<T, C>
where
    T: KNNFloat,
    C: Eq + Hash + Clone + Debug,
{
    pub metrics: Distance,
    pub x_train: Option<Array2<T>>,
    pub y_train: Option<Array1<C>>,
    pub k: Option<usize>,
}

impl<T, C> KNNClassifier<T, C>
where
    T: KNNFloat,
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

    /// Stores the features (Array2<T>) and the targets (Array1<C>)
    pub fn fit(&mut self, x: Array2<T>, y: Array1<C>, k: usize) -> Result<(), DataError> {
        // CORRECTION: validation_shape_2d_1d expects an Array2 and an Array1.
        // Array1<C> should implement T:Float, which it doesn't.
        // The same validation cannot be reused for C.
        // A simpler function is needed to validate x:Array2<T> and y:Array1<C>
        // I'm using the simple validation body for the demo:
        if x.shape()[0] != y.shape()[0] {
            return Err(DataError::DimensionMismatch(x.shape()[0], y.shape()[0]));
        }
        if x.iter().any(|&e| e.is_nan()) {
            return Err(DataError::NaNData);
        }

        self.x_train = Some(x);
        self.y_train = Some(y);
        self.k = Some(k);

        Ok(())
    }

    /// Makes predictions for a new dataset (x_test)
    pub fn predict(&self, x_test: &Array2<T>) -> Result<Array1<C>, DataError> {
        // Correction: take x_test by reference
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
        let mut predictions = Array1::uninit(x_test.shape()[0]);

        // Loop through each test sample
        for i in 0..x_test.shape()[0] {
            let test_sample = x_test.row(i);

            // --- 2. Calculating Distances (Lógica idéntica al Regressor) ---

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

            let mut indexed_distances: Vec<(T, usize)> = distances_array
                .into_iter()
                .enumerate()
                .map(|(index, dist)| (dist.clone(), index))
                .collect();

            indexed_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let limit = k_val.min(n_train);

            let k_nearest_indices: Vec<usize> = indexed_distances
                .into_iter()
                .take(limit)
                .map(|(_, index)| index)
                .collect();

            // --- 4. Classification Vote (Mode) ---

            let mut class_counts: HashMap<C, usize> = HashMap::new();

            for &index in k_nearest_indices.iter() {
                let class = y_train[[index]].clone();
                *class_counts.entry(class).or_insert(0) += 1;
            }

            // Determine the majority class (the one with max count)
            let predicted_class = class_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(class, _)| class)
                .ok_or(DataError::EmptyData)?; // Correction: Si la liste des voisins est vide (impossible si n_train > 0)

            // 5. Store the prediction
            // CORRECTION: Utiliser le unsafe .assume_init() après avoir assigné une valeur
            unsafe {
                predictions.as_slice_mut().unwrap()[i] = predicted_class.clone();
            }
        }

        // Finalisation : Convertir l'Array uninitialized en Array initialisé
        Ok(unsafe { predictions.assume_init() })
    }
}
