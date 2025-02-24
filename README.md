
# 🧠 Random-ML: A Randomized Machine Learning Framework

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg) 
![License](https://img.shields.io/badge/License-MIT-green.svg) 
![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)  
 
Random-ML is a **high-performance machine learning library** that implements **randomized neural networks** and **functional link architectures**.  
It provides fast, accurate, and scalable models like **Random Vector Functional Link (RVFL)**, **Extreme Learning Machines (ELM)**, and **Stochastic Configuration Networks (SCN)**.  

🚀 **Key Features**
- **Randomized Neural Networks** → Avoids backpropagation, leading to **faster training**.
- **Supports Ensemble Learning** → Includes **Boosting & Bagging**.
- **Multiple Activation Functions** → `relu`, `sigmoid`, `tanh`, `leaky_relu`, `sin`.
- **Works with Scikit-Learn** → Seamlessly integrates into the existing ML ecosystem.
- **Customizable Models** → Allows tuning **hidden units, weight initialization, and regularization**.

---

## ⚡ **Installation**
You can install Random-ML via pip:
```sh
pip install random-ml
```
or install from source:
```sh
git clone https://github.com/yourusername/random-ml.git
cd random-ml
pip install -e .
```

---

## 🛠 **Usage Examples**
### 🚀 **1. Regression with RVFL**
```python
from random_ml.regressor import RVFLRegressor
import numpy as np

# Generate sample data
X = np.random.rand(100, 10)
y = np.sin(X[:, 0])  # Regression target

# Train RVFL model
rvfl_reg = RVFLRegressor(in_dim=10, n_hidden_units=50, activation="relu")
rvfl_reg.fit(X, y)
y_pred = rvfl_reg.predict(X)

print("Predictions:", y_pred[:5])
```

---

### 🔥 **2. Classification with RVFL**
```python
from random_ml.classifier import RVFLClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate classification data
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RVFL classifier
rvfl_cls = RVFLClassifier(in_dim=2, n_hidden_units=50, activation="relu")
rvfl_cls.fit(X_train, y_train)
y_pred = rvfl_cls.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print(f"RVFL Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
```

---

### 🤖 **3. Using Boosting & Bagging for Classification**
```python
from random_ml.ensemble import RVFLBoostingClassifier, RVFLBaggingClassifier

rvfl_kwargs = {"n_hidden_units": 50, "activation": "relu"}

# AdaBoost with RVFL
boosting_cls = RVFLBoostingClassifier(in_dim=2, rvfl_kwargs=rvfl_kwargs, n_estimators=20, random_state=42)
boosting_cls.fit(X_train, y_train)
y_pred_boosting = boosting_cls.predict(X_test)

# Bagging with RVFL
bagging_cls = RVFLBaggingClassifier(in_dim=2, rvfl_kwargs=rvfl_kwargs, n_estimators=20, random_state=42)
bagging_cls.fit(X_train, y_train)
y_pred_bagging = bagging_cls.predict(X_test)

print(f"Boosting Accuracy: {accuracy_score(y_test, y_pred_boosting) * 100:.2f}%")
print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred_bagging) * 100:.2f}%")
```

---

## 📖 **Documentation & API Reference**
Full documentation is available at [ReadTheDocs](https://random-ml.readthedocs.io/).

To build the documentation locally:
```sh
cd docs
make html
```
Then open `docs/build/html/index.html` in your browser.

### 📄 **Available Modules**
| Module             | Description |
|--------------------|------------|
| `random_ml.rvfl`  | Implements RVFL-based models |
| `random_ml.ensemble` | Includes Boosting & Bagging implementations |
| `randomml.classifier` | Provides RVFL-based classifiers |
| `random_ml.regressor` | Contains RVFL-based regressors |
| `randomml.mlpedrvfl` | Implements the MLPedRVFL model |

---

## 🤝 **Contributing**
We welcome contributions to improve Random-ML! To contribute:
1. **Fork the repository**.
2. **Clone your fork**:
   ```sh
   git clone https://github.com/yourusername/random-ml.git
   cd random-ml
   ```
3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
4. **Create a new branch**:
   ```sh
   git checkout -b feature_branch
   ```
5. **Make changes & commit**:
   ```sh
   git add .
   git commit -m "Describe your changes"
   ```
6. **Push to GitHub & submit a pull request**.

For details, see [Contributing Guide](contribute.md).

---

## 🏆 **Citing random-ml**
The preprint is Submitted to Pattern Recognition Letter, please cite it:

Vinay Kumar Giri, Rahul Goswami, Vimlesh Kumar, Synergistic Regression through MLP and edRVFL Fusion: The MLPedRVFL Model for Enhanced Performance and Efficiency, Preprint Submitted to Pattern Recognition Letter.

### **BibTeX Citation**
```bibtex
@misc{randomml2024,
  A placeholder for bibTeX citation
}
```

Coming Soon - The paper is under review. Yaay!

---

## 📝 **License**
Random-ML is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---
## 🎯 **What's Next?**
- ✅ Implement more **activation functions**
- ✅ Improve **training speed**
- ✅ Support **GPU acceleration**
- ✅ Add **new ensemble methods**
- ✅ Open to community contributions!

🔥 **Now you're ready to use Random-ML for fast and scalable machine learning!** 🚀  
Let us know if you need help or have feature requests! 😊✨  
```

