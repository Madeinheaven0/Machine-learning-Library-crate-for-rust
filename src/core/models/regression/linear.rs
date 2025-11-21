use crate::core::metrics::regression::metrics::EvaluationMetrics;
use crate::errors::DataError;
use ndarray::linalg::Dot;
use ndarray::{Array1, Array2, ArrayBase, Dim, LinalgScalar, OwnedRepr, ScalarOperand};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::num_traits::real::Real;
use num_traits::{Float, FromPrimitive, One, Zero};
use std::fmt::Debug;

pub struct LinearRegression<T>
where
    T: Float + FromPrimitive + LinalgScalar + Dot<T, Output = Array1<T>> + Debug + ScalarOperand + 'static + Zero + One,
{
    pub weights: Option<Array1<T>>,
    pub bias: Option<f64>,
}

impl<T> LinearRegression<T>
where
    T: Float + FromPrimitive + LinalgScalar + Dot<T, Output = Array1<T>> + Debug + ScalarOperand + 'static,
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
    ) -> Result<(), DataError>
    where
        T: Float + FromPrimitive + LinalgScalar + Dot<T, Output = Array1<T>> + Debug,
        // T::Output of Dot<T> is Array1<T> for x.dot(&weights)
        for<'a> &'a ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>:
            Dot<&'a Array1<T>, Output = Array1<T>>,
        for<'a> &'a ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>: Dot<&'a Array1<T>>,
        // Make sure that x.t() can also endow.
        for<'a> ArrayBase<ndarray::ViewRepr<&'a T>, Dim<[usize; 2]>>:
            Dot<&'a Array1<T>, Output = Array1<T>>,
    {
        // --- 1. Data Validation ---
        if x.is_empty() {
            return Err(DataError::EmptyData);
        }

        if y.is_empty() {
            return Err(DataError::EmptyTarget);
        }

        if x.iter().any(|&e| e.is_nan()) {
            return Err(DataError::NaNData);
        }

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let y_len = y.shape()[0];

        if n_samples != y_len {
            return Err(DataError::DimensionMismatch(n_samples, y_len));
        }

        // --- 2. Parameter Initialization ---

        let dist = Normal::new(T::from(0.0).unwrap(), T::from(1.0).unwrap()).unwrap();
        let mut weights = Array1::random(n_features, dist);
        let mut bias = T::from(0.0).unwrap(); // The bias is a scalar T (not an Array1)

        // Conversion of scalars to type T
        let n_samples_t = T::from(n_samples).ok_or(DataError::InvalidData)?;
        let lr_t = T::from(lr).ok_or(DataError::InvalidData)?;

        // --- 3. Training Loop (Gradient Descent) ---
        for iteration in 0..n_iter {
            // Prediction : X.W + b
            // .dot is a T, and Array + Scalar addition is handled by ndarray
            let prediction = x.dot(&weights) + bias;

            // Error : (Y_pred - Y_reel)
            let error = &prediction - y;

            // Gradients
            // Gradient of weights : X.T * Erreur / N
            let weight_gradient = x.t().dot(&error) / n_samples_t;

            // Gradient of bias : Mean of the error
            let bias_gradient = error.sum() / n_samples_t;

            // Update : W = W - dW * LR
            //Use lr_t, and the operations are between Array1<T> and Scalar T
            weights = &weights - &(&weight_gradient * lr_t);
            bias = bias - bias_gradient * lr_t;

            if iteration % 1000 == 0 {
                // Compute the loss (MSE)
                let loss = error.mapv(|e| e * e).sum() / n_samples_t;
                println!("Iteration {}, Loss: {:.6?}", iteration, loss);
            }
        }

        // --- 4. Storage ---
        self.weights = Some(weights);
        self.bias = Some(bias);

        Ok(())
    }

    pub fn evaluate(&self, x_test: &Array2<T>, y_test: &Array1<T>) -> EvaluationMetrics<T> {
        let predictions = self.predict(x_test);
        let n_samples = T::from_usize(y_test.len()).ok_or(DataError::InvalidData)?;

        // Mean Squared Error
        let mse = predictions.predictions.iter() // Utilisez le champ Array1<T> interne
            .zip(y_test.iter());

        // R-squared
        let y_mean = y_test.mean().unwrap_or(T::zero());
        let tss = y_test.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
        let rss = predictions
            .iter()
            .zip(y_test.iter())
            .map(|(pred, actual)| (actual - pred).powi(2))
            .sum::<f64>();
        let r_squared = if tss == T::zero() { T::one() } else { T::one() - (rss / tss) };

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
            (Some(weights), Some(bias)) => Predictions {
                predictions: x.dot(weights) + bias as T,
            },
            _ => panic!("Model not fitted. Call fit() first."),
        }
    }
}

pub struct Predictions<T> {
    predictions: Array1<T>,
}
