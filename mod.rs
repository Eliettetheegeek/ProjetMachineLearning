// src/data_utils/mod.rs

use nalgebra::{DVector};

/// calcule l'errreur quadratique moyenne
/// MSE = (1/N) * sum((targets - predictions)^2)
pub fn mean_squared_error(predictions: &DVector<f64>, targets: &DVector<f64>) -> f64 {
    if predictions.len() != targets.len() {
        panic!("Les vecteurs de prédictions et de cibles doivent avoir la même taille.");
    }

    // Calcul de la différence au carré (y - y_hat)^2
    let squared_diff = (targets - predictions).map(|x| x * x);

    // Somme des différences au carré, divisée par le nombre d'échantillons (N)
    squared_diff.sum() / (targets.len() as f64)
}

/// fonction d'activation sigmoide (c pour le PMC)
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Dérivée de la Sigmoïde (utile pour la Backpropagation)
pub fn sigmoid_derivative(y: f64) -> f64 {
    // y est l'output de la sigmoïde (sigmoid(x))
    y * (1.0 - y)
}