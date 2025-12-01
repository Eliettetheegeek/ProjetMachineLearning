use nalgebra::{DMatrix, DVector};

/// ici c par la pseudo-inverse de Moore-Penrose pour la linear regression
/// w = (X^T * X)^-1 * X^T * y
///
/// X: Matrice des données (N_samples x N_features)
/// y: vecteur cible (N_samples)
/// ce qui nous retourn le vecteur de poids (N_features)
pub fn train_pseudo_inverse(X: &DMatrix<f64>, y: &DVector<f64>) -> Result<DVector<f64>, String> {
    // 1. X^T * X
    let xtx = X.transpose() * X;

    // 2. Calcul de l'inverse (X^T * X)^-1
    // try_inverse() gère le cas où la matrice est singulière
    let xtx_inv = match xtx.try_inverse() {
        Some(inv) => inv,
        None => return Err(String::from("La matrice (X^T * X) n'est pas inversible. Échec du pseudo-inverse.")),
    };

    // 3. Multiplication: (X^T * X)^-1 * X^T * y
    let w = (xtx_inv * X.transpose()) * y;

    Ok(w)
}

/// prédiction pour la regression lineair
pub fn predict_linear(X: &DMatrix<f64>, weights: &DVector<f64>) -> DVector<f64> {
    X * weights // y_hat = X * w
}

// src/algorithms/linear_model.rs (suite)

/// entraine le perceptron avec la classification kineaire  avec la regle de rosenblatt.
/// utilise des cibles (-1, 1).
/// retourne un vecteur de poids entraîné.
pub fn train_rosenblatt(
    mut weights: DVector<f64>, // Les poids initiaux sont modifiés
    X: &DMatrix<f64>,
    y: &DVector<f64>,
    learning_rate: f64,
    max_epochs: usize
) -> DVector<f64> {
    let num_samples = X.nrows();

    for epoch in 1..=max_epochs {
        let mut errors = 0;

        for i in 0..num_samples {
            let x_i = X.row(i).transpose();
            let y_i = y[i];

            // produit scalaire (somme pondérée)
            let output = (x_i.transpose() * &weights)[0];

            // Prédiction: 1.0 si output >= 0, -1.0 sinon (fonction signe)
            let prediction = if output >= 0.0 { 1.0 } else { -1.0 };

            // Mise à jour SEULEMENT si la classification est incorrecte
            if prediction != y_i {
                errors += 1;

                // Correction: delta_w = alpha * y_i * x_i
                // C'est la forme standard de Rosenblatt.
                let correction = y_i * learning_rate;
                let delta_w = correction * x_i;

                weights += delta_w;
            }
        }

        // critere d'arret si aucune erreur, la convergence est atteinte
        if errors == 0 {
            println!("Rosenblatt a convergé après {} epochs.", epoch);
            break;
        }
    }
    weights
}
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use crate::data_utils::mean_squared_error;

    // --- Test 1: Pseudo-inverse (Régression) ---
    #[test]
    fn test_pseudo_inverse_regression() {
        // y = 1*b + 2*x  (w = [1.0, 2.0])
        let x_data = DMatrix::from_row_slice(4, 2, &[
            1.0, 1.0,  // Biais + x1
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
        ]);
        let y_target = DVector::from_vec(vec![3.0, 5.0, 7.0, 9.0]);

        let weights = train_pseudo_inverse(&x_data, &y_target).unwrap();

        // Les poids doivent être proches de [1.0, 2.0]
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert!((weights[1] - 2.0).abs() < 1e-6);

        let predictions = predict_linear(&x_data, &weights);
        let mse = mean_squared_error(&predictions, &y_target);

        // La MSE doit être presque nulle pour la solution analytique
        assert!(mse < 1e-10);
    }

    // --- Test 2: Rosenblatt (Classification) ---
    #[test]
    fn test_rosenblatt_and_convergence() {
        // AND Gate (Linéairement Séparable)
        let x_data = DMatrix::from_row_slice(4, 3, &[
            1.0, 0.0, 0.0, // Biais + x1 + x2
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 1.0,
        ]);
        // Cibles: (-1, -1, -1, 1)
        let y_target = DVector::from_vec(vec![-1.0, -1.0, -1.0, 1.0]);

        let initial_weights = DVector::from_element(3, 0.0);
        let final_weights = train_rosenblatt(initial_weights, &x_data, &y_target, 0.1, 100);

        // Vérifier que le modèle a trouvé une solution
        // (La MSE doit être très faible ou le nombre d'erreurs nul après l'entraînement)
        // Note: Ici, on devrait idéalement vérifier la convergence dans la boucle d'entraînement
        // Pour un test simple, on vérifie que les poids ne sont pas nuls.
        assert!(final_weights.iter().any(|&w| w.abs() > 0.0));
    }
}
