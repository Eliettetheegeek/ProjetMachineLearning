import os
import pandas as pd

# Chemin vers votre dossier principal d'images
DATA_DIR = 'dataset_amphibiens'
OUTPUT_CSV = 'amphibiens_map.csv'

data_list = []

# Parcourir les sous-dossiers (chaque sous-dossier est une classe)
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)

    # S'assurer que c'est bien un dossier et qu'il n'est pas vide
    if os.path.isdir(class_path):
        # Parcourir les fichiers dans le dossier de classe
        for filename in os.listdir(class_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')): # Filtrer les images
                # Chemin relatif complet à l'image
                full_path = os.path.join(class_path, filename)

                # Enregistrer le chemin d'accès et le label
                data_list.append({
                    'filepath': full_path,
                    'label': class_name,
                    'label_numeric': 0 # Vous ajouterez la conversion numérique plus tard (0, 1, 2...)
                })

# Créer un DataFrame Pandas et l'enregistrer en CSV
df = pd.DataFrame(data_list)

# Attribuer des indices numériques (0, 1, 2...) aux classes
# C'est l'encodage nécessaire pour les algorithmes ML
class_to_id = {name: i for i, name in enumerate(df['label'].unique())}
df['label_numeric'] = df['label'].map(class_to_id)


# Enregistrer le fichier de mappage
df.to_csv(OUTPUT_CSV, index=False)

print(f"Fichier de mappage généré: {OUTPUT_CSV}")
print(f"Classes et encodage numérique: {class_to_id}")

# Afficher les 5 premières lignes du résultat:
print("\nPremières lignes du CSV de mappage:")
print(df.head())