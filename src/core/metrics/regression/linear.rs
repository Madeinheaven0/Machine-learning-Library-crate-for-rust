use super::metrics::RegressionMetric;
use crate::errors::DataError;
use ndarray::Array1;
use ndarray::prelude::*;

fn mse<T>(x: Array1<T>, y: Array1<T>) -> Result<T, DataError> {
    if x.len() != y.len() {
        return Err(DataError::DimensionMismatch(x.shape()[0], y.shape()[0]));
    }

    if x.is_empty() || y.is_empty() {
        return Err(DataError::EmptyData);
    }

    let mut x_nan;

    let mut y_nan;

    x.is_nan().map(|x| match x {
        false => x_nan += 0.,
        true => x_nan += 1,
    });

    y.is_nan().map(|x| match x {
        false => y_nan += 0,
        true => y_nan += 1,
    });

    if x_nan > 0 || y_nan > 0 {
        return Err(DataError::NaNData);
    }

    let n_samples = x.shape()[0] as f64;

    let error = (x - y).powi(2);

    Ok(error.sum() / n_samples)
}

fn rmse(x: Array1<f64>, y: Array1<f64>) -> Result<f64, DataError> {
    if x.len() != y.len() {
        return Err(DataError::DimensionMismatch(x.shape()[0], y.shape()[0]));
    }

    if x.is_empty() || y.is_empty() {
        return Err(DataError::EmptyData);
    }

    let mut x_nan;
    let mut y_nan;

    x.is_nan().map(|x| match x {
        false => x_nan += 0.,
        true => x_nan += 1,
    });

    y.is_nan().map(|x| match x {
        false => y_nan += 0,
        true => y_nan += 1,
    });

    if x_nan > 0 || y_nan > 0 {
        return Err(DataError::NaNData);
    }

    let n_samples = x.shape()[0] as f64;

    let error = (x - y).powi(2);

    Ok((error.sum() / n_samples).sqrt())
}

fn lmse(x: Array1<f64>, y: Array1<f64>) -> Result<f64, DataError> {
    if x.len() != y.len() {
        return Err(DataError::DimensionMismatch(x.shape()[0], y.shape()[0]));
    }

    if x.is_empty() || y.is_empty() {
        return Err(DataError::EmptyData);
    }

    let mut x_nan;
    let mut y_nan;

    x.is_nan().map(|x| match x {
        false => x_nan += 0.,
        true => x_nan += 1,
    });

    y.is_nan().map(|x| match x {
        false => y_nan += 0,
        true => y_nan += 1,
    });

    if x_nan > 0 || y_nan > 0 {
        return Err(DataError::NaNData);
    }

    let n_samples = x.shape()[0] as f64;

    let error = (x - y).powi(2);
    Ok((error.sum() / n_samples).ln())
}

fn lrmse<T>(x: Array1<T>, y: Array1<T>) -> Result<f64, DataError> {
    if x.len() != y.len() {
        return Err(DataError::DimensionMismatch(x.shape()[0], y.shape()[0]));
    }

    if x.is_empty() || y.is_empty() {
        return Err(DataError::EmptyData);
    }

    let mut x_nan;
    let mut y_nan;

    x.is_nan().map(|x| match x {
        false => x_nan += 0.,
        true => x_nan += 1,
    });

    y.is_nan().map(|x| match x {
        false => y_nan += 0,
        true => y_nan += 1,
    });

    if x_nan > 0 || y_nan > 0 {
        return Err(DataError::NaNData);
    }

    let n_samples = x.shape()[0] as f64;

    let error = (x - y).powi(2);

    Ok((error.sum() / n_samples).sqrt().ln())
}

fn mean_absolute_error<T>(x: Array1<T>, y: Array1<T>) -> Result<T, DataError> {
    if x.len() != y.len() {
        return Err(DataError::DimensionMismatch(x.shape()[0], y.shape()[0]));
    }

    if x.is_empty() || y.is_empty() {
        return Err(DataError::EmptyData);
    }

    let mut x_nan;
    let mut y_nan;

    x.is_nan().map(|x| match x {
        false => x_nan += 0.,
        true => x_nan += 1,
    });

    y.is_nan().map(|x| match x {
        false => y_nan += 0,
        true => y_nan += 1,
    });

    if x_nan > 0 || y_nan > 0 {
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
        RegressionMetric::MeanSquareError => mse(x, y)?,
        RegressionMetric::MeanAbsoluteError => mean_absolute_error(x, y)?,
        RegressionMetric::RootMeanSquaredError => rmse(x, y)?,
        RegressionMetric::LogMeanSquaredError => lmse(x, y)?,
        RegressionMetric::LogRootMeanSquaredError => lrmse(x, y)?,
    };

    Ok(measure)
}
