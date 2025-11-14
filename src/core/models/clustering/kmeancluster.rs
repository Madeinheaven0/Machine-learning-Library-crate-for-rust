use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, Sub, Div};

use crate::errors::DataError;
use crate::core::metrics::regression::metrics::Distance;

pub struct KMeans<T>
where
    T: Float + Clone + Debug + PartialOrd + PartialEq,
{
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: T,
    pub distance_metric: Distance,
    pub centroids: Option<Array2<T>>, // Centroid matrix (K x n_features)
}

impl<T> KMeans<T>
where
    T: Float
    + Add
    + Sub
    + Div
    + PartialOrd
    + PartialEq
    + Clone
    + Debug,
{
    pub fn new(k: usize, max_iterations: usize, tolerance: T, metric: Distance) -> Self {
        Self {
            k,
            max_iterations,
            tolerance,
            distance_metric: metric,
            centroids: None,
        }
    }

    /// Fonction pour initialiser les centroïdes aléatoirement
    fn initialize_centroids(&self, x: &Array2<T>) -> Array2<T> {
        // In a real-world implementation, this would require a random number generator.

        // To simplify, we take the first K samples.

        // This is the easiest way to obtain K valid rows.
        let mut centroids = Array2::zeros((self.k, x.shape()[1]));

        for i in 0..self.k {
            if i < x.shape()[0] {
                centroids.row_mut(i).assign(&x.row(i));
            } else {

            }
        }
        centroids
    }

    pub fn fit(&mut self, x: Array2<T>) -> Result<Array1<usize>, DataError> {
        // Basic validation (a K-Means requires at least K points)
        if x.shape()[0] < self.k || self.k == 0 {
            return Err(DataError::DimensionMismatch(x.shape()[0], self.k));
        }

        // 1. Initialization of centroids
        let mut centroids = self.initialize_centroids(&x);
        let n_samples = x.shape()[0];
        let mut labels = Array1::zeros(n_samples); // Vector to store the cluster assignment

        for _ in 0..self.max_iterations {
            let mut new_centroids = Array2::zeros(centroids.dim());
            let mut cluster_counts = Array1::zeros(self.k);
            let mut max_centroid_movement = T::zero();

            // 2. Assignment (E-Step: Expectation)
            for i in 0..n_samples {
                let sample = x.row(i);
                let mut min_dist = T::max_value();
                let mut closest_centroid_index = 0;

                for (j, centroid) in centroids.outer_iter().enumerate() {
                    // Distance calculation (Same as KNN)
                    let dist = match self.distance_metric {
                        Distance::Euclidean => {
                            let diff = &sample - &centroid;
                            // For K-Means, it suffices to minimize the squared distance
                            diff.map(|v| v.powi(2)).sum()
                        },
                        Distance::Manhattan => {
                            let diff = &sample - &centroid;
                            diff.map(|v| v.abs()).sum()
                        },
                        _ => return Err(DataError::InvalidMetric),
                    };

                    if dist < min_dist {
                        min_dist = dist;
                        closest_centroid_index = j;
                    }
                }

                // Update the cluster assignment
                labels[[i]] = closest_centroid_index;

                // Add the point to the future new centroid
                new_centroids.row_mut(closest_centroid_index).scaled_add(T::one(), &sample);
                cluster_counts[[closest_centroid_index]] = cluster_counts[[closest_centroid_index]] + T::one();
            }

            // 3. Centroid Update (M-Step: Maximization)
            for j in 0..self.k {
                let count = cluster_counts[[j]];
                if count > T::zero() {
                    let old_centroid = centroids.row(j).to_owned();
                    let mut new_centroid = new_centroids.row_mut(j);

                    // Calculation of the new average (new_centroid / count)
                    new_centroid.map_mut(|v| *v = *v / count);

                    // Calculate the centroid's motion for convergence
                    let movement = (&new_centroid - &old_centroid)
                        .map(|v| v.abs())
                        .sum();

                    if movement > max_centroid_movement {
                        max_centroid_movement = movement;
                    }
                }
            }

            // Update the centroids for the next iteration
            centroids.assign(&new_centroids);

            // 4. Stopping criterion (Convergence)
            if max_centroid_movement < self.tolerance {
                break;
            }
        }

        self.centroids = Some(centroids);
        Ok(labels)
    }

    pub fn compute_inertia(
        &self,
        x: &Array2<T>,
        labels: &Array1<usize>,
    ) -> Result<T, DataError> {

        let centroids = self.centroids.as_ref().ok_or(DataError::NoteFittedModel)?;
        let mut total_inertia = T::zero();

        // Checking dimensions
        if x.shape()[0] != labels.len() {
            return Err(DataError::DimensionMismatch(x.shape()[0], self.k));
        }

        // Iterate over each sample
        for i in 0..x.shape()[0] {
            let sample = x.row(i);
            let cluster_index = labels[[i]];

            // Ensure that the cluster index is valid
            if cluster_index >= self.k {
                return Err(DataError::DimensionMismatch(cluster_index, self.k));
            }

            let centroid = centroids.row(cluster_index);

            // Calculating the square of the distance between the sample and its centroid
            // (The square of the Euclidean distance is the standard measure for inertia)
            let squared_distance = match self.distance_metric {
                Distance::Euclidean => {
                    let diff = &sample - &centroid;
                    // Note: This is the square of the distance (L2 norm squared)
                    diff.map(|v| v.powi(2)).sum()
                },
                // Note: This is the square of the distance (L² norm // If the metric is not Euclidean, we still use
                // the squared error (Euclidean distance squared) as a measure
                // of inertia, because that is the standard definition).
                _ => {
                    let diff = &sample - &centroid;
                    diff.map(|v| v.powi(2)).sum()
                },
            };

            // Add to the total inertia
            total_inertia = total_inertia + squared_distance;
        }

        Ok(total_inertia)
    }
}
