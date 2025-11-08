use crate::core::metrics::regression::metrics::EvaluationMetrics;
use crate::errors::DataError;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::num_traits::real::Real;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

pub struct LinearRegression<T>
where
    T: Add + Sub + Mul + Div + Rem + Neg + PartialOrd + PartialEq + Clone + Debug,
{
    pub weights: Option<Array1<T>>,
    pub bias: Option<f64>,
}

impl<T: Add + Sub + Mul + Div + Rem + Neg + PartialOrd + PartialEq + Clone + Debug>
    LinearRegression<T>
{
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: None,
        }
    }

    pub fn fit(
        &mut self,
        x: &Array2<T>,
        y: &Array1<T>,
        n_iter: i64,
        lr: f64,
    ) -> Result<(), DataError> {
        if x.is_empty() {
            return Err(DataError::EmptyData);
        }

        if y.is_empty() {
            return Err(DataError::EmptyTarget);
        }

        if x.is_nan() {
            return Err(DataError::NaNData);
        }

        let n_samples = x.shape()[0] as f64;
        let y_len = y.shape()[0] as f64;

        if n_samples != y_len {
            // return Err(DataError::DimensionMismatch(ndarray::ShapeError))
            return Err(DataError::DimensionMismatch(
                n_samples as usize,
                y_len as usize,
            ));
        }

        // Initialisation
        let mut weights = Array1::random(&n_samples, Normal::new(0., 1.).unwrap());
        let mut bias = Array1::zeros(&n_samples);

        for iteration in 0..n_iter {
            let prediction = x.dot(&weights) + &bias;

            // Erreur
            let error = &prediction - y;

            // Gradients
            let weight_gradient = x.t().dot(&error) / n_samples;
            let bias_gradient = error / n_samples;

            // Mise à jour
            weights = &weights - &(&weight_gradient * lr);
            bias -= bias_gradient * lr;

            if iteration % 1000 == 0 {
                let loss = error.mapv(|e| e * e).sum() / n_samples;
                println!("Iteration {}, Loss: {:.6}", iteration, loss);
            }
        }

        self.weights = Some(weights);
        self.bias = Some(bias);

        Ok(())
    }

    pub fn evaluate(&self, x_test: &Array2<T>, y_test: &Array1<T>) -> EvaluationMetrics<T> {
        let predictions = self.predict(x_test);
        let n_samples = y_test.len() as T;

        // Mean Squared Error
        let mse = predictions
            .iter()
            .zip(y_test.iter())
            .map(|(pred, actual)| (pred - actual).powi(2))
            .sum::<f64>()
            / &n_samples;

        // R-squared
        let y_mean = y_test.mean().unwrap_or(0.0);
        let tss = y_test.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
        let rss = predictions
            .iter()
            .zip(y_test.iter())
            .map(|(pred, actual)| (actual - pred).powi(2))
            .sum::<f64>();
        let r_squared = if tss == 0.0 { 1.0 } else { 1.0 - (rss / tss) };

        // Mean Absolute Error
        let mae = predictions
            .iter()
            .zip(y_test.iter())
            .map(|(pred, actual)| (pred - actual).abs())
            .sum::<f64>()
            / n_samples;

        EvaluationMetrics {
            mse,
            r_squared,
            mae,
        }
    }

    pub fn predict(&self, x: &Array2<T>) -> Predictions<T> {
        match (&self.weights, self.bias) {
            (Some(weights), Some(bias)) => Predictions{predictions: x.dot(weights) + bias},
            _ => panic!("Model not fitted. Call fit() first."),
        }
    }
}


pub struct Predictions<T> {
    predictions: Array1<T>
}