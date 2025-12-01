// src/lib.rs

// --- 1. Déclaration des Modules ---
// Nécessaire pour que le compilateur sache que ces dossiers existent
pub mod data_utils;
pub mod algorithms;

// --- 2. Exports (pour une utilisation interne simplifiée) ---
pub use algorithms::linear_model::{train_pseudo_inverse, train_rosenblatt, predict_linear};
pub use algorithms::mlp::MLP;
pub use data_utils::mean_squared_error;

// --- 3. Dépendances & Imports des Fonctions Wrappers ---
use pyo3::prelude::*;
use nalgebra::{DMatrix, DVector};

// IMPORTANT : Cette ligne doit être ICI, AVANT d'être utilisée dans #[pymodule]
use algorithms::mlp::train_mlp_py;


// --- 4. Fonctions Wrappers Définies Localement ---
// La fonction rust_mse doit être définie ici, avant #[pymodule]
#[pyfunction]
fn rust_mse(py_predictions: Vec<f64>, py_targets: Vec<f64>) -> PyResult<f64> {
    // Conversion de Vec<f64> Python vers DVector<f64> nalgebra
    let predictions = DVector::from_vec(py_predictions);
    let targets = DVector::from_vec(py_targets);

    // Appel de votre fonction Rust implémentée
    Ok(data_utils::mean_squared_error(&predictions, &targets))
}


// --- 5. Déclaration du Module Python (UN SEUL BLOC) ---

#[pymodule]
pub fn ml_lib(_py: Python, m: &PyModule) -> PyResult<()> {

    // Ajout de la fonction MSE
    m.add_function(wrap_pyfunction!(rust_mse, m)?)?;

    // Ajout de la fonction PMC (qui était "unresolved")
    m.add_function(wrap_pyfunction!(train_mlp_py, m)?)?;

    // Ajoutez ici d'autres fonctions (Pseudo-inverse, Rosenblatt) si vous avez leurs wrappers
    // m.add_function(wrap_pyfunction!(rust_train_rosenblatt, m)?)?;

    Ok(())
}
