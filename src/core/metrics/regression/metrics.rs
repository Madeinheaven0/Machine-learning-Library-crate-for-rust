pub enum RegressionMetric {
    MeanSquareError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    LogMeanSquaredError,
    LogRootMeanSquaredError,
}

pub enum Distance {
    Euclidean,
    Manhattan,
    Minkowski,
}

#[derive(Debug, Clone)]
pub struct EvaluationMetrics<T> {
    pub mse: T,
    pub r_squared: T,
    pub mae: T,
}

impl<T: std::fmt::Display> EvaluationMetrics<T> {
    pub fn print(&self) {
        println!("Evaluation Metrics:");
        println!("  MSE: {:.6}", self.mse);
        println!("  R²:  {:.6}", self.r_squared);
        println!("  MAE: {:.6}", self.mae);
    }
}
