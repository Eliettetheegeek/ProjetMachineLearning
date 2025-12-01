# run_plumbing_test.py
import pandas as pd
import numpy as np
import ml_lib # Le nom de votre package dans Cargo.toml

# Les données de test
predictions = [1.0, 2.0, 3.0]
targets = [1.5, 2.5, 3.5]

# Appel de la fonction Rust exposée
mse_result = ml_lib.rust_mse(predictions, targets)

print(f"Prédictions: {predictions}")
print(f"Cibles: {targets}")
print(f"MSE calculée par Rust: {mse_result}")
# Résultat attendu: 0.25