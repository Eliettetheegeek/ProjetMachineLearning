# Dans votre script client_python/run_tests.py

from PIL import Image
import numpy as np
import pandas as pd
import ml_lib # Votre librairie Rust

# Fonction de Prétraitement
def image_to_feature_vector(filepath, size=(32, 32)):
    # 1. Ouvrir l'image
    img = Image.open(filepath)

    # 2. Convertir en niveaux de gris (simplifie les features : 1 canal au lieu de 3)
    img = img.convert('L')

    # 3. Redimensionner l'image à la taille cible (ex: 32x32)
    img = img.resize(size)

    # 4. Convertir en tableau NumPy (pixels de 0 à 255)
    np_array = np.array(img, dtype=np.float64)

    # 5. Normalisation des pixels (ramener les valeurs entre 0 et 1)
    np_array /= 255.0

    # 6. Aplatir la matrice 32x32 en un vecteur 1024
    return np_array.flatten().tolist()

# Charger le fichier de mappage
df = pd.read_csv('amphibiens_map.csv')

# Séparer un petit échantillon (Ex: les 15 premières images)
X_features = []
Y_targets = []

for index, row in df.head(15).iterrows(): # Utilisez un petit échantillon pour tester
    # Transformation de l'image en vecteur
    features = image_to_feature_vector(row['filepath'])
    X_features.extend(features)
    Y_targets.append(row['label_numeric'])

num_rows = len(Y_targets)
num_cols = len(X_features) // num_rows # Doit être 1024 dans notre exemple 32x32

print(f"Dataset prêt : {num_rows} échantillons, {num_cols} features.")