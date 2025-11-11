import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class LinearModel:
    """Version Python pure pour comparaison avec Rust"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W = None
        self.loss_history = []
    
    def predict(self, X):
        if self.W is None:
            raise ValueError("Modèle non entraîné")
        scores = X @ self.W
        return np.sign(scores)
    
    def fit(self, X, y, n_iterations=10000):
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones(n_samples), X]
        
        if self.W is None:
            self.W = np.random.randn(n_features + 1, 1) * 0.01
        
        print(f"Entraînement Python: {n_samples} samples, {n_iterations} iterations")
        
        for it in tqdm(range(n_iterations), desc="Training Python"):
            k = np.random.randint(0, n_samples)
            X_k = X_bias[k:k+1].T
            y_k = y[k]
            
            y_pred = 1.0 if (self.W.T @ X_k)[0, 0] >= 0 else -1.0
            
            if y_pred != y_k:
                self.W += self.learning_rate * (y_k - y_pred) * X_k
            
            if it % 1000 == 0:
                predictions = self.predict(X_bias)
                accuracy = np.mean(predictions.flatten() == y)
                self.loss_history.append(1 - accuracy)

class MultiClassLinear:
    """Version multi-classes du modèle linéaire (One-vs-Rest)"""
    
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.models = []
        for _ in range(n_classes):
            model = LinearModel(learning_rate=learning_rate)
            model.W = np.random.randn(n_features + 1, 1) * 0.01
            self.models.append(model)
        self.n_classes = n_classes
    
    def fit(self, X, y, n_iterations=10000):
        print(f"Entraînement multi-classes sur {X.shape[0]} échantillons...")
        
        for class_id in range(self.n_classes):
            print(f"  Classe {class_id} vs Rest...")
            binary_y = np.where(y == class_id, 1.0, -1.0)
            self.models[class_id].fit(X, binary_y, n_iterations)
    
    def predict(self, X):
        X_bias = np.c_[np.ones(len(X)), X]
        scores = []
        
        for model in self.models:
            scores.append(model.predict(X_bias).flatten())
        
        scores = np.array(scores).T
        return np.argmax(scores, axis=1)

def extract_features(image_path, target_size=(64, 64)):
    """Extrait des features simples des images pour le modèle linéaire"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB').resize(target_size)
            img_array = np.array(img)
            
            mean_color = img_array.mean(axis=(0, 1)) / 255.0
            std_color = img_array.std(axis=(0, 1)) / 255.0
            
            hist_r = np.histogram(img_array[:,:,0], bins=5, range=(0,255))[0] / (target_size[0]*target_size[1])
            hist_g = np.histogram(img_array[:,:,1], bins=5, range=(0,255))[0] / (target_size[0]*target_size[1])
            hist_b = np.histogram(img_array[:,:,2], bins=5, range=(0,255))[0] / (target_size[0]*target_size[1])
            
            features = np.concatenate([mean_color, std_color, hist_r, hist_g, hist_b])
            return features
            
    except Exception as e:
        print(f"Erreur avec {image_path}: {e}")
        return None

def load_dataset(dataset_path="dataset_amphibiens"):
    """Charge le dataset et extrait les features"""
    X = []
    y = []
    class_mapping = {"grenouille": 0, "crapaud": 1, "têtard": 2}
    
    for class_name, class_id in class_mapping.items():
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"❌ Dossier manquant: {class_path}")
            continue
            
        print(f"Traitement de la classe: {class_name}")
        count = 0
        
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                features = extract_features(image_path)
                
                if features is not None:
                    X.append(features)
                    y.append(class_id)
                    count += 1
        
        print(f"  → {count} images traitées")
    
    if len(X) == 0:
        print("❌ Aucune image trouvée - vérifiez le chemin du dataset")
        return np.array([]), np.array([]), class_mapping
    
    return np.array(X), np.array(y), class_mapping

def generate_linearly_separable_data(n_samples=100, random_state=42):
    """Génère des données linéairement séparables"""
    np.random.seed(random_state)
    X = np.random.random((n_samples, 2)) * 2 - 1
    
    y = np.where(0.5 * X[:, 0] + X[:, 1] - 0.2 >= 0, 1.0, -1.0)
    
    return X, y

def generate_xor_data(n_samples=200):
    """Génère le dataset XOR"""
    np.random.seed(42)
    X = []
    y = []
    
    for _ in range(n_samples // 4):
        X.append(np.random.randn(2) * 0.1 + [0.2, 0.2])
        y.append(-1.0)
        
        X.append(np.random.randn(2) * 0.1 + [0.2, 0.8])
        y.append(1.0)
        
        X.append(np.random.randn(2) * 0.1 + [0.8, 0.2])
        y.append(1.0)
        
        X.append(np.random.randn(2) * 0.1 + [0.8, 0.8])
        y.append(-1.0)
    
    return np.array(X), np.array(y)

def test_linear_separable_python():
    """TEST 1: Données linéairement séparables (Python)"""
    print("\n" + "="*60)
    print("TEST 1: LINÉAIREMENT SÉPARABLE (PYTHON)")
    print("="*60)
    
    X, y = generate_linearly_separable_data(n_samples=150)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolor='black', alpha=0.7)
    plt.title('Données d\'entraînement')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.3)
    
    model_py = LinearModel(learning_rate=0.001)
    model_py.fit(X, y, n_iterations=20000)
    
    X_bias = np.c_[np.ones(len(X)), X]
    predictions = model_py.predict(X_bias)
    accuracy = np.mean(predictions.flatten() == y)
    
    print(f"\n📊 Résultats Python:")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Poids finaux: {model_py.W.flatten()}")
    
    plt.subplot(1, 2, 2)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_bias = np.c_[np.ones(len(grid_points)), grid_points]
    Z = model_py.predict(grid_bias).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['#ffcccc', '#ccccff'], alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolor='black', alpha=0.7)
    
    if model_py.W[2] != 0:
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(model_py.W[0] + model_py.W[1] * x_line) / model_py.W[2]
        plt.plot(x_line, y_line, 'g-', linewidth=2, label='Frontière')
    
    plt.title(f'Frontière de décision\nAccuracy: {accuracy:.1%}')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return accuracy, model_py

def test_xor_problem():
    """TEST 2: Problème XOR - limites du linéaire"""
    print("\n" + "="*60)
    print("TEST 2: PROBLÈME XOR (LIMITES LINÉAIRES)")
    print("="*60)
    
    X, y = generate_xor_data(n_samples=200)
    
    model_py = LinearModel(learning_rate=0.0001)
    model_py.fit(X, y, n_iterations=30000)
    
    X_bias = np.c_[np.ones(len(X)), X]
    predictions = model_py.predict(X_bias)
    accuracy = np.mean(predictions.flatten() == y)
    
    print(f"\n📊 Performances sur XOR:")
    print(f"   Accuracy: {accuracy:.3f} (attendu ~0.5 pour aléatoire)")
    print(f"   Le modèle linéaire échoue sur XOR - OK ✅")
    
    plt.figure(figsize=(8, 6))
    
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_bias = np.c_[np.ones(len(grid_points)), grid_points]
    Z = model_py.predict(grid_bias).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['#ffcccc', '#ccccff'], alpha=0.6)
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolor='black', alpha=0.8)
    plt.title(f'Modèle Linéaire sur XOR\nAccuracy: {accuracy:.1%} ❌', fontweight='bold')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return accuracy

def test_convergence_study():
    """TEST 3: Étude de la convergence"""
    print("\n" + "="*60)
    print("TEST 3: ÉTUDE DE CONVERGENCE")
    print("="*60)
    
    X, y = generate_linearly_separable_data(n_samples=100)
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    final_accuracies = []
    
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        model = LinearModel(learning_rate=lr)
        
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones(n_samples), X]
        model.W = np.random.randn(n_features + 1, 1) * 0.01
        
        accuracies = []
        
        for it in range(5000):
            k = np.random.randint(0, n_samples)
            X_k = X_bias[k:k+1].T
            y_k = y[k]
            
            y_pred = 1.0 if (model.W.T @ X_k)[0, 0] >= 0 else -1.0
            
            if y_pred != y_k:
                model.W += lr * (y_k - y_pred) * X_k
            
            if it % 100 == 0:
                predictions = model.predict(X_bias)
                accuracy = np.mean(predictions.flatten() == y)
                accuracies.append(accuracy)
        
        final_accuracies.append(accuracies[-1])
        
        plt.subplot(2, 2, i+1)
        plt.plot(accuracies)
        plt.title(f'LR = {lr}\nFinal: {accuracies[-1]:.3f}')
        plt.xlabel('Itérations (x100)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📈 Résultats convergence:")
    for lr, acc in zip(learning_rates, final_accuracies):
        print(f"   LR={lr}: Accuracy finale = {acc:.3f}")
    
    return learning_rates, final_accuracies

def test_real_dataset():
    """TEST 4: Dataset réel grenouille/crapaud/têtard"""
    print("\n" + "="*70)
    print("TEST RÉEL : DATASET GRENOUILLE/CRAPAUD/TÊTARD")
    print("="*70)
    
    X, y, class_mapping = load_dataset()
    
    if len(X) == 0:
        print("❌ Aucune donnée chargée - création du dataset nécessaire")
        print("💡 Exécutez d'abord votre script de collecte d'images!")
        return None, None, None, None
    
    print(f"\n📊 Dataset chargé:")
    print(f"   {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"   Répartition des classes: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = MultiClassLinear(n_features=X.shape[1], n_classes=3, learning_rate=0.001)
    model.fit(X_train, y_train, n_iterations=5000)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    
    print(f"\n📈 Performances:")
    print(f"   Accuracy entraînement: {train_acc:.3f}")
    print(f"   Accuracy test: {test_acc:.3f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=list(class_mapping.keys()))
    disp.plot(cmap='Blues')
    plt.title("Matrice de confusion - Modèle Linéaire")
    plt.tight_layout()
    plt.show()
    
    # Analyse des erreurs
    print(f"\n🔍 Analyse des erreurs:")
    for i, class_name in enumerate(class_mapping.keys()):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(test_pred[class_mask] == y_test[class_mask])
            print(f"   {class_name}: {class_acc:.3f}")
    
    return model, X, y, class_mapping

def run_all_tests():
    print("\n" + "🎯 VALIDATION MODÈLE LINÉAIRE ".center(70, "="))
    
    # Test 1: Linéairement séparable (artificiel)
    acc1, model1 = test_linear_separable_python()
    
    # Test 2: XOR 
    acc2 = test_xor_problem()
    
    # Test 3: Convergence
    lrs, accs = test_convergence_study()
    
    # Test 4: Dataset réel
    model_real, X_real, y_real, classes = test_real_dataset()
    
    # Résumé final
    print("\n" + "="*70)
    print(" RÉSUMÉ COMPLET - ÉTAPE 2")
    print("="*70)
    print(f" Test 1 (Linéaire artificiel): {acc1:.3f}")
    print(f" Test 2 (XOR - limites):       {acc2:.3f}") 
    print(f" Test 3 (Convergence):         Complété")
    
    if model_real is not None:
        train_pred = model_real.predict(X_real)
        real_acc = np.mean(train_pred == y_real)
        print(f" Test 4 (Dataset réel):        {real_acc:.3f}")
    else:
        print(f" Test 4 (Dataset réel):        Dataset manquant")
    
    print("="*70)
    print("\n Pour améliorer les performances sur le dataset réel:")
    print("   • Plus de données d'entraînement")
    print("   • Features plus sophistiquées (SIFT, HOG, etc.)")
    print("   • Modèles non-linéaires (PMC, RBF)")
    print("   • Augmentation des données")

if __name__ == "__main__":
    run_all_tests()