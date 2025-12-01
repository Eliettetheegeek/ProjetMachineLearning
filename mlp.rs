use nalgebra::{DMatrix, DVector};
use pyo3::{pyfunction, PyErr, PyResult};
use crate::data_utils::{sigmoid, sigmoid_derivative, mean_squared_error}; // Import des utilitaires

/// structure représentant un pmc.
pub struct MLP {
    // chaque élément represente une couche à partir de la première couche cachée
    pub weights: Vec<DMatrix<f64>>,
    pub biases: Vec<DVector<f64>>,
}

impl MLP {
    pub fn new(layers: &[usize]) -> Self {
        if layers.len() < 2 {
            panic!("Un MLP doit avoir au moins une couche d'entrée et une couche de sortie.");
        }

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..(layers.len() - 1) {
            let rows = layers[i+1]; // Nbr de neurones de la couche suivante
            let cols = layers[i];   // Nbr de neurones de la couche actuelle

            // Initialisation aléatoire des poids entre -1 et 1
            let w = DMatrix::new_random(rows, cols) * 2.0 - DMatrix::from_element(rows, cols, 1.0);
            weights.push(w);

            // Initialisation aléatoire des biais
            let b = DVector::new_random(rows) * 2.0 - DVector::from_element(rows, 1.0);
            biases.push(b);
        }

        MLP { weights, biases }
    }

    /// propagation Avant (Forward Pass)
    /// retourne les outputs (activations) de chaque couche (Z_i et A_i).
    fn forward(&self, input: &DVector<f64>) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let mut z_values = Vec::new(); // Sommes pondérées (Z = W * A + B)
        let mut a_values = vec![input.clone()]; // Outputs (Activations), A[0] est l'input

        let mut current_activation = input.clone();

        for i in 0..self.weights.len() {
            // 1. Calcul de Z: W * A + B
            let z = &self.weights[i] * &current_activation + &self.biases[i];
            z_values.push(z.clone());

            // 2. Calcul de A: Sigmoid(Z)
            let a = z.map(|x| sigmoid(x));
            a_values.push(a.clone());

            current_activation = a;
        }

        (z_values, a_values)
    }

    /// retropropagation (Backpropagation) pour un seul échantillon (SGD)
    /// calcule le gradient de la MSE par rapport aux poids et biais.
    fn backpropagate(
        &self,
        target: &DVector<f64>,
        _z_values: &[DVector<f64>],
        a_values: &[DVector<f64>]
    ) -> (Vec<DMatrix<f64>>, Vec<DVector<f64>>) {
        let num_layers = self.weights.len();

        // Initialisation des gradients (mêmes dimensions que les poids/biais)
        let mut dw = vec![DMatrix::zeros(0, 0); num_layers];
        let mut db = vec![DVector::zeros(0); num_layers];

        // --- 1. Couche de Sortie (L) ---

        // Erreur de la couche de sortie: delta_L = (A_L - Y) .* sigmoid_prime(Z_L)
        // Note: La dérivée de la MSE par rapport à A_L est 2 * (A_L - Y)
        let mut delta = (a_values.last().unwrap() - target).scale(2.0).component_mul(
            &a_values.last().unwrap().map(|y| sigmoid_derivative(y))
        );

        db[num_layers - 1] = delta.clone();
        // dw_L = delta_L * A_{L-1}^T
        dw[num_layers - 1] = &delta * a_values[num_layers - 1].transpose();

        // --- 2. Couches cachées (l = L-1 jusqu'à 1) ---
        for l in (0..num_layers - 1).rev() {
            // delta_l = (W_{l+1}^T * delta_{l+1}) .* sigmoid_prime(Z_l)
            delta = (self.weights[l + 1].transpose() * &delta).component_mul(
                &a_values[l + 1].map(|y| sigmoid_derivative(y))
            );

            db[l] = delta.clone();
            // dw_l = delta_l * A_{l}^T
            dw[l] = &delta * a_values[l].transpose();
        }

        (dw, db)
    }

    /// entrainement utilisant la descente de gradient stochastique (SGD)
    pub fn train_sgd(&mut self, X: &DMatrix<f64>, y: &DVector<f64>, learning_rate: f64, epochs: usize) {
        let num_samples = X.nrows();

        for epoch in 1..=epochs {
            let mut total_loss = 0.0;

            for i in 0..num_samples {
                let x_i = X.row(i).transpose();
                //faut qu'on s'assure que y_i a le bon format pour la sortie (e.g., DVector<f64> de taille 1)
                let y_i = DVector::from_element(1, y[i]);

                // 1. Propagation Avant
                let (z_values, a_values) = self.forward(&x_i);
                let output = a_values.last().unwrap();

                // 2. Calcul du gradient (retropropagation)
                let (dw, db) = self.backpropagate(&y_i, &z_values, &a_values);

                // 3. Mise à jour des poids (SGD)
                for l in 0..self.weights.len() {
                    self.weights[l] = &self.weights[l] - dw[l].scale(learning_rate);
                    self.biases[l] = &self.biases[l] - db[l].scale(learning_rate);
                }

                // 4. Calcul de la perte pour suivi
                total_loss += mean_squared_error(output, &y_i);
            }

            println!("Epoch {} - MSE: {}", epoch, total_loss / (num_samples as f64));
        }
    }
}
// src/algorithms/mlp.rs (suite)

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use crate::data_utils::mean_squared_error;

    // Fonction utilitaire pour la prédiction sur un dataset complet
    fn predict_mlp(model: &MLP, X: &DMatrix<f64>) -> DVector<f64> {
        let mut predictions = DVector::zeros(X.nrows());
        for i in 0..X.nrows() {
            let input = X.row(i).transpose();
            // Le forward pass retourne une activation finale de DVector<f64>
            let (_, activations) = model.forward(&input);
            // On prend la première (et seule) valeur de l'output layer pour la prédiction
            predictions[i] = activations.last().unwrap()[0];
        }
        predictions
    }

    // --- Test 3: PMC (SGD) sur XOR (Non Linéaire) ---
    #[test]
    fn test_mlp_xor_convergence() {
        // Données XOR : non linéairement séparables, ce qui échouerait pour Rosenblatt.
        // Input (avec Biais implicite pour simplifier l'exemple, ou 2 features)
        let x_data = DMatrix::from_row_slice(4, 2, &[
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ]);
        // Cibles: (0, 1, 1, 0)
        let y_target = DVector::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        // 1. Initialisation du modèle
        // PMC avec 2 entrées, 3 neurones cachés, 1 sortie (2-3-1)
        let mut mlp = MLP::new(&[2, 3, 1]);

        // Calcul de la MSE initiale (avant entraînement)
        let initial_predictions = predict_mlp(&mlp, &x_data);
        let initial_mse = mean_squared_error(&initial_predictions, &y_target);

        println!("MSE initiale (XOR): {}", initial_mse);

        // 2. Entraînement
        let learning_rate = 0.5;
        let epochs = 5000;

        // L'entraînement utilise la fonction train_sgd que vous avez implémentée
        // (Elle affiche la MSE à chaque epoch pour le suivi)
        mlp.train_sgd(&x_data, &y_target, learning_rate, epochs);

        // 3. Validation de la convergence
        let final_predictions = predict_mlp(&mlp, &x_data);
        let final_mse = mean_squared_error(&final_predictions, &y_target);

        println!("MSE finale (XOR): {}", final_mse);

        // La MSE finale doit être significativement inférieure à l'initiale
        // et basse (par exemple, moins de 0.01)
        assert!(final_mse < initial_mse);
        assert!(final_mse < 0.01, "La MSE finale ({}) est trop élevée, le modèle n'a pas convergé correctement.", final_mse);

        // 4. Test de la classification finale (pour s'assurer de la bonne prédiction)
        // Vérifie si la prédiction est correcte à 90% (arbitraire)
        let mut correct_predictions = 0;
        for i in 0..final_predictions.len() {
            // Seuil de 0.5 pour la classification binaire
            let predicted_class = if final_predictions[i] >= 0.5 { 1.0 } else { 0.0 };
            if predicted_class == y_target[i] {
                correct_predictions += 1;
            }
        }
        let accuracy = correct_predictions as f64 / final_predictions.len() as f64;
        println!("Précision finale (XOR): {}%", accuracy * 100.0);

        // S'assurer d'une haute précision (e.g., 90% sur ce cas simple)
        assert!(accuracy >= 0.90, "Le modèle n'a pas atteint une précision suffisante ({}%).", accuracy * 100.0);
    }
}
#[pyfunction]
pub fn train_mlp_py(
    data_flat: Vec<f64>,
    target_flat: Vec<f64>,
    num_rows: usize,
    num_cols: usize,
    hidden_layer_size: usize, // Taille de la couche cachée (ex: 10 neurones)
    learning_rate: f64,
    epochs: usize,
) -> PyResult<Vec<Vec<f64>>> {

    if data_flat.len() != num_rows * num_cols {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "La taille du vecteur de données ne correspond pas aux dimensions fournies."
        ));
    }

    // 1. Conversion des données d'entrée (X) en DMatrix
    // Utilisation de from_row_slice pour lire ligne par ligne (comme NumPy)
    let X = DMatrix::from_row_slice(num_rows, num_cols, &data_flat);

    // 2. Conversion de la cible (Y) en DVector
    let Y = DVector::from_vec(target_flat);

    // 3. Déterminer la taille de la couche de sortie (nombre de classes)
    // Pour la classification d'images, cela sera 3 (crapaud, grenouille, têtard)
    let num_output_classes = Y.iter().map(|v| *v as usize).max().unwrap_or(0) + 1;

    // 4. Initialisation du PMC
    let input_size = num_cols;
    let topology = vec![input_size, hidden_layer_size, num_output_classes];
    let mut mlp = MLP::new(&topology);

    // 5. Entraînement du modèle (appel de votre fonction Rust implémentée)
    mlp.train_sgd(&X, &Y, learning_rate, epochs);

    // 6. Extraction et renvoi des poids sous forme de Vec<Vec<f64>>
    let final_weights: Vec<Vec<f64>> = mlp.weights.iter()
        .map(|matrix| matrix.data.as_vec().clone())
        .collect();

    Ok(final_weights)
}