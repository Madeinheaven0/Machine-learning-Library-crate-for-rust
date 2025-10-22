use ndarray::{Array2, ArrayD, Axis, Array1};
use crate::core::traits::StatisticsOps;
use crate::errors::DataError;

impl StatisticsOps for Array2<f64> {
    fn mean(&self) -> Result<ArrayD<f64>, DataError> {
        if self.is_empty() {
            return Err(DataError::EmptyData);
        }
        Ok(self.mean_axis(Axis(0))
            .ok_or(DataError::EmptyData)?
            .into_dyn())
    }

    fn std(&self) -> Result<ArrayD<f64>, DataError> {
        if self.is_empty() {
            return Err(DataError::EmptyData);
        }
        Ok(self.std_axis(Axis(0), 0.0).into_dyn())
    }

    fn min(&self) -> Result<ArrayD<f64>, DataError> {
        if self.is_empty() {
            return Err(DataError::EmptyData);
        }
        Ok(self.fold_axis(Axis(0), f64::INFINITY, |acc, x| acc.min(*x)).into_dyn())
    }

    fn max(&self) -> Result<ArrayD<f64>, DataError> {
        if self.is_empty() {
            return Err(DataError::EmptyData);
        }
        Ok(self.fold_axis(Axis(0), f64::NEG_INFINITY, |acc, x| acc.max(*x)).into_dyn())
    }

    fn median(&self) -> Result<ArrayD<f64>, DataError> {
        calculate_quantile(self, 0.5)
    }

    fn first_quantile(&self) -> Result<ArrayD<f64>, DataError> {
        calculate_quantile(self, 0.25)
    }

    fn third_quantile(&self) -> Result<ArrayD<f64>, DataError> {
        calculate_quantile(self, 0.75)
    }
}

// FONCTION CORRIGÉE - ÇA MARCHE PUTAIN
fn calculate_quantile(arr: &Array2<f64>, q: f64) -> Result<ArrayD<f64>, DataError> {
    if arr.is_empty() {
        return Err(DataError::EmptyData);
    }

    let n_cols = arr.ncols();
    let mut results = Vec::with_capacity(n_cols);

    for col_idx in 0..n_cols {
        let mut col_data: Vec<f64> = arr.column(col_idx)
            .iter()
            .copied()
            .filter(|x| !x.is_nan())
            .collect();

        if col_data.is_empty() {
            return Err(DataError::CalculationError("Column contains only NaN values".to_string()));
        }

        col_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = col_data.len();
        let index = (q * (n - 1) as f64).round() as usize;
        let index = index.min(n - 1);

        results.push(col_data[index]);
    }

    // CONVERSION FINALE QUI MARCHE
    let array_1d = Array1::from(results);
    Ok(array_1d.into_dyn())
}