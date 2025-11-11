use pyo3::prelude::*;
use rand::prelude::*;

#[pyclass]
pub struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
}

#[pymethods]
impl LinearModel {
    #[new]
    pub fn new(n_features: usize) -> Self {
        let mut rng = thread_rng();
        LinearModel {
            weights: (0..n_features).map(|_| rng.gen_range(-0.1..0.1)).collect(),
            bias: 0.0,
        }
    }

    pub fn fit(&mut self, xs: Vec<Vec<f64>>, ys: Vec<f64>, epochs: usize, lr: f64) {
        let n = ys.len() as f64;
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

            for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
                *w -= lr * (g / n);
            }
            self.bias -= lr * (grad_b / n);

            if epoch % 1000 == 0 {
                println!("Epoch {}/{}", epoch, epochs);
            }
        }
    }

    pub fn predict(&self, x: Vec<f64>) -> f64 {
        self.weights.iter().zip(x.iter())
            .map(|(w, xi)| w * xi).sum::<f64>() + self.bias
    }
}

// ✅ Nouvelle version compatible avec PyO3 v0.22+
#[pymodule]
fn projetannuel(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LinearModel>()?;
    Ok(())
}
