use crate::core::metrics::regression::metrics::Distance;
use crate::errors::DataError;
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::Debug;

// Composite trait for the  K-Means
pub trait KMeansFloat:
    Float + FromPrimitive + Clone + Debug + PartialOrd + PartialEq + Zero
{
}
impl KMeansFloat for f32 {}
impl KMeansFloat for f64 {}

pub struct KMeans<T>
where
    T: KMeansFloat,
{
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: T,
    pub distance_metric: Distance,
    pub centroids: Option<Array2<T>>, // Centroids matrices (K x n_features)
}

impl<T> KMeans<T>
where
    T: KMeansFloat,
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

    /// Initializes the centroids by taking the first K samples (simplified)
    fn initialize_centroids(&self, x: &Array2<T>) -> Array2<T> {
        let mut centroids = Array2::zeros((self.k, x.shape()[1]));

        for i in 0..self.k {
            if i < x.shape()[0] {
                centroids.row_mut(i).assign(&x.row(i));
            }
        }
        centroids
    }

    pub fn fit(&mut self, x: Array2<T>) -> Result<Array1<usize>, DataError> {
        if x.shape()[0] < self.k || self.k == 0 {
            return Err(DataError::DimensionMismatch(x.shape()[0], self.k));
        }

        // 1. Initialization
        let mut centroids = self.initialize_centroids(&x);
        let n_samples = x.shape()[0];
        let mut labels = Array1::zeros(n_samples);

        for _ in 0..self.max_iterations {
            let mut new_centroids_sum = Array2::zeros(centroids.dim()); // Sum of points for the future centroid
            let mut cluster_counts = Array1::zeros(self.k); // Counters (usize)
            let mut max_centroid_movement = T::zero();
            let mut next_centroids = centroids.clone();
            let mut changed = false;

            // 2. Assignment (E-Step)
            for i in 0..n_samples {
                let sample = x.row(i);
                let mut min_dist = T::max_value();
                let mut closest_centroid_index = 0;

                for (j, centroid) in centroids.outer_iter().enumerate() {
                    // Calcul de la distance
                    let dist = match self.distance_metric {
                        Distance::Euclidean => {
                            let diff = &sample - &centroid;
                            diff.mapv(|v| v.powi(2)).sum()
                        }
                        Distance::Manhattan => {
                            let diff = &sample - &centroid;
                            diff.mapv(|v| v.abs()).sum()
                        }
                        _ => return Err(DataError::InvalidMetric),
                    };

                    if dist < min_dist {
                        min_dist = dist;
                        closest_centroid_index = j;
                    }
                }

                labels[[i]] = closest_centroid_index;

                // Accumulation of points in new_centroids_sum
                new_centroids_sum
                    .row_mut(closest_centroid_index)
                    .scaled_add(T::one(), &sample);
                cluster_counts[[closest_centroid_index]] += 1;
            }

            // 3. Centroids (M-Step) Update
            for j in 0..self.k {
                let count_usize = cluster_counts[[j]];

                if count_usize > 0 {
                    let count_t = T::from_usize(count_usize).ok_or(DataError::InvalidData)?;

                    let old_centroid = centroids.row(j).to_owned();
                    let mut new_centroid_row = next_centroids.row_mut(j);

                    // Calculating the new average and assigning it
                    new_centroid_row.assign(&new_centroids_sum.row(j).mapv(|v| v / count_t));
                    changed = true;

                    // Calculating the motion
                    let movement = (&new_centroid_row.to_owned() - &old_centroid)
                        .map(|v| v.abs())
                        .sum();

                    if movement > max_centroid_movement {
                        max_centroid_movement = movement;
                    }
                }
                // If count_usize == 0, the centroid remains in its previous position
            }

            // Centroid update for the next iteration
            if changed {
                centroids.assign(&next_centroids);
            }

            // 4. Stopping criterion
            if max_centroid_movement < self.tolerance {
                break;
            }
        }

        self.centroids = Some(centroids);
        Ok(labels)
    }

    pub fn compute_inertia(&self, x: &Array2<T>, labels: &Array1<usize>) -> Result<T, DataError> {
        let centroids = self.centroids.as_ref().ok_or(DataError::NoteFittedModel)?;
        let mut total_inertia = T::zero();

        if x.shape()[0] != labels.len() {
            return Err(DataError::DimensionMismatch(x.shape()[0], labels.len()));
        }

        for i in 0..x.shape()[0] {
            let sample = x.row(i);
            let cluster_index = labels[[i]];

            if cluster_index >= self.k {
                return Err(DataError::DimensionMismatch(cluster_index, self.k));
            }

            let centroid = centroids.row(cluster_index);

            //Inertia is the sum of the squares of the EUCLIDEAN distances
            let diff = &sample - &centroid;
            let squared_distance: T = diff.mapv(|v| v.powi(2)).sum();

            total_inertia = total_inertia + squared_distance;
        }

        Ok(total_inertia)
    }
}
