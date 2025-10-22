use ndarray::Array2;
use crate::core::traits::{MatrixOps, TensorLike};
use crate::errors::DataError;

impl MatrixOps for Array2<f64> {
    fn transpose(&self) -> Result<Box<dyn TensorLike<Elem = Self::Elem>>, DataError> {
        // .t() crée une vue transposée ; .to_owned() crée une nouvelle Array2.
        let transposed = self.t().to_owned();

        // Retourne la nouvelle Array2 empaquetée dans Box<dyn TensorLike>
        Ok(Box::new(transposed))
    }

    fn matmul(&self, other: &dyn TensorLike<Elem = Self::Elem>) -> Result<Box<dyn TensorLike<Elem = Self::Elem>>, DataError> {
        // 1. Assurez-vous que l'autre tenseur est un ArrayD
        let other_array_d = other.to_array()?;

        // 2. Tenter de le convertir en Array2 pour l'opération matricielle
        let other_mat: Array2<f64> = other_array_d
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| DataError::DimensionMismatch("L'opérande droite (B) doit être 2D pour le produit matriciel.".to_string()))?;

        // 3. Effectuer le produit matriciel (dot product)
        let result_mat = self.dot(&other_mat);

        // 4. Retourner le résultat empaqueté
        Ok(Box::new(result_mat))
    }
}