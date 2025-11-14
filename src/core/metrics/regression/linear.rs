use super::metrics::RegressionMetric;
use crate::core::validation::*;
use crate::errors::DataError;
use ndarray::Array1;
use ndarray::prelude::*;

pub fn mse<T>(x: &Array1<T>, y: &Array1<T>) -> Result<T, DataError> {
    validation_mix(x, y)?;

    let n_samples = x.shape()[0] as f64;

    let error = (x - y).powi(2);

    Ok(error.sum() / n_samples)
}

pub fn rmse(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, DataError> {
    validation_mix(x, y)?;

    let n_samples = x.shape()[0] as f64;

    let error = (x - y).powi(2);

    Ok((error.sum() / n_samples).sqrt())
}

pub fn lmse(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, DataError> {
    validation_mix(x, y)?;

    let n_samples = x.shape()[0] as f64;

    let error = (x - y).powi(2);
    Ok((error.sum() / n_samples).ln())
}

pub fn lrmse<T>(x: &Array1<T>, y: &Array1<T>) -> Result<f64, DataError> {
    validation_mix(x, y)?;

    let n_samples = x.shape()[0] as f64;

    let error = (x - y).powi(2);

    Ok((error.sum() / n_samples).sqrt().ln())
}

pub fn mean_absolute_error<T>(x: &Array1<T>, y: &Array1<T>) -> Result<T, DataError> {
    if x.iter().any(|&e| e.is_nan()) || y.iter().any(|&e| e.is_nan()) {
        return Err(DataError::NaNData);
    }

    let n_samples = x.shape()[0] as f64;

    Ok((x - y).abs().sum() / n_samples)
}

pub fn regression_metric<T>(
    x: Array1<T>,
    y: Array1<T>,
    metric: RegressionMetric,
) -> Result<T, DataError> {
    let measure = match metric {
        RegressionMetric::MeanSquareError => mse(&x, &y)?,
        RegressionMetric::MeanAbsoluteError => mean_absolute_error(&x, &y)?,
        RegressionMetric::RootMeanSquaredError => rmse(&x, &y)?,
        RegressionMetric::LogMeanSquaredError => lmse(&x, &y)?,
        RegressionMetric::LogRootMeanSquaredError => lrmse(&x, &y)?,
    };

    Ok(measure)
}
