use rand::prelude::*;
use std::fs::File;
use std::io::Write;

#[derive(Debug)]
pub struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearModel {
    pub fn new(n_features: usize) -> Self {
        let mut rng = thread_rng();
        LinearModel {
            weights: (0..n_features).map(|_| rng.gen_range(-0.1..0.1)).collect(),
            bias: 0.0,
        }
    }

    pub fn fit(&mut self, xs: &[Vec<f64>], ys: &[f64], epochs: usize, lr: f64) {
        let n = ys.len() as f64;
        println!("🎯 Entraînement sur {} échantillons, {} epochs", xs.len(), epochs);

        for epoch in 0..epochs {
            let mut grad_w = vec![0.0; self.weights.len()];
            let mut grad_b = 0.0;

            for (xrow, &y) in xs.iter().zip(ys.iter()) {
                let pred: f64 = self.weights.iter().zip(xrow.iter())
                    .map(|(w, xi)| w * xi).sum::<f64>() + self.bias;
                let err = pred - y;

                for (g, xi) in grad_w.iter_mut().zip(xrow.iter()) {
                    *g += err * xi;
                }
                grad_b += err;
            }

            // Mise à jour des poids
            for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
                *w -= lr * (g / n);
            }
            self.bias -= lr * (grad_b / n);

            // Affichage de progression
            if epoch % 1000 == 0 {
                let current_loss = self.calculate_loss(xs, ys);
                println!("Epoch {}/{} - Loss: {:.4}", epoch, epochs, current_loss);
            }
        }

        let final_loss = self.calculate_loss(xs, ys);
        println!("✅ Entraînement terminé - Loss finale: {:.4}", final_loss);
    }

    fn calculate_loss(&self, xs: &[Vec<f64>], ys: &[f64]) -> f64 {
        let mut total_loss = 0.0;
        for (x, &y) in xs.iter().zip(ys.iter()) {
            let pred = self.predict(x);
            let err = pred - y;
            total_loss += err * err;
        }
        total_loss / xs.len() as f64
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.weights.iter().zip(x.iter())
            .map(|(w, xi)| w * xi).sum::<f64>() + self.bias
    }

    pub fn accuracy(&self, xs: &[Vec<f64>], ys: &[f64]) -> f64 {
        let mut correct = 0;
        for (x, &y_true) in xs.iter().zip(ys.iter()) {
            let y_pred = self.predict(x);
            // Classification binaire : signe de la prédiction
            if (y_pred >= 0.0 && y_true == 1.0) || (y_pred < 0.0 && y_true == -1.0) {
                correct += 1;
            }
        }
        correct as f64 / xs.len() as f64
    }
}

fn generate_linearly_separable_data(n_samples: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = thread_rng();
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for _ in 0..n_samples {
        let x1: f64 = rng.gen_range(-1.0..1.0);
        let x2: f64 = rng.gen_range(-1.0..1.0);
        let y = if 0.5 * x1 + x2 - 0.2 > 0.0 { 1.0 } else { -1.0 };

        xs.push(vec![x1, x2]);
        ys.push(y);
    }
    (xs, ys)
}

fn generate_xor_data(n_samples: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = thread_rng();
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for _ in 0..n_samples/4 {
        // Cluster 1: bas-gauche → -1
        xs.push(vec![rng.gen_range(0.1..0.4), rng.gen_range(0.1..0.4)]);
        ys.push(-1.0);

        // Cluster 2: bas-droit → +1
        xs.push(vec![rng.gen_range(0.6..0.9), rng.gen_range(0.1..0.4)]);
        ys.push(1.0);

        // Cluster 3: haut-gauche → +1
        xs.push(vec![rng.gen_range(0.1..0.4), rng.gen_range(0.6..0.9)]);
        ys.push(1.0);

        // Cluster 4: haut-droit → -1
        xs.push(vec![rng.gen_range(0.6..0.9), rng.gen_range(0.6..0.9)]);
        ys.push(-1.0);
    }
    (xs, ys)
}

fn main() {
    println!("🎯 TESTS MODÈLE LINÉAIRE RUST");
    println!("{}", "=".repeat(50));

    // TEST 1: Données linéairement séparables
    println!("\n📊 TEST 1: Données Linéairement Séparables");
    let (xs_linear, ys_linear) = generate_linearly_separable_data(100);
    let mut model_linear = LinearModel::new(2);
    model_linear.fit(&xs_linear, &ys_linear, 5000, 0.01);

    let accuracy_linear = model_linear.accuracy(&xs_linear, &ys_linear);
    println!("✅ Accuracy: {:.1}%", accuracy_linear * 100.0);

    // Sauvegarde pour visualisation
    let mut file = File::create("test_linear.csv").unwrap();
    writeln!(file, "x1,x2,y_true,y_pred").unwrap();
    for (x, &y_true) in xs_linear.iter().zip(ys_linear.iter()) {
        let y_pred = model_linear.predict(x);
        writeln!(file, "{:.4},{:.4},{:.4},{:.4}", x[0], x[1], y_true, y_pred).unwrap();
    }

    // TEST 2: Problème XOR
    println!("\n📊 TEST 2: Problème XOR");
    let (xs_xor, ys_xor) = generate_xor_data(200);
    let mut model_xor = LinearModel::new(2);
    model_xor.fit(&xs_xor, &ys_xor, 5000, 0.01);

    let accuracy_xor = model_xor.accuracy(&xs_xor, &ys_xor);
    println!("✅ Accuracy: {:.1}% (attendu ~50%)", accuracy_xor * 100.0);

    let mut file = File::create("test_xor.csv").unwrap();
    writeln!(file, "x1,x2,y_true,y_pred").unwrap();
    for (x, &y_true) in xs_xor.iter().zip(ys_xor.iter()) {
        let y_pred = model_xor.predict(x);
        writeln!(file, "{:.4},{:.4},{:.4},{:.4}", x[0], x[1], y_true, y_pred).unwrap();
    }

    println!("\n{}", "=".repeat(50));
    println!("📁 Fichiers générés:");
    println!("   • test_linear.csv");
    println!("   • test_xor.csv");
    println!("\n🎨 Pour visualiser: python python/visualization.py");
}