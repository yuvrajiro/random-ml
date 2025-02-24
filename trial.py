# # Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import importlib
from sklearn.preprocessing import MinMaxScaler
import randomml
importlib.reload(randomml)
from randomml import RVFLRegressor, RVFLClassifier
#
# # --------------------------
# # 🚀 REGRESSION TEST (SINE CURVE)
# # --------------------------
# # Generate sine wave data
# X_reg = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
# y_reg = np.sin(X_reg).ravel()
#
# scaler = MinMaxScaler()
# X_reg = scaler.fit_transform(X_reg)
# # Split into train and test sets
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
#
# # Initialize and train RVFL regressor
# regressor = RVFLRegressor(in_dim=1, n_nodes=500, direct_link=True, alpha=None)
# regressor.fit(X_train_reg, y_train_reg)
#
# # Predict on test data
# y_pred_reg = regressor.predict(X_test_reg)
#
# # Plot results
# plt.figure(figsize=(8, 5))
# plt.scatter(X_test_reg, y_test_reg, color="blue", label="True Sine Wave")
# plt.scatter(X_test_reg, y_pred_reg, color="red", label="RVFL Predictions", marker="x")
# plt.plot(X_reg, np.sin(X_reg), "g--", alpha=0.6)  # Ground truth curve
# plt.legend()
# plt.title("RVFL Regression on Sine Curve")
# plt.xlabel("X")
# plt.ylabel("sin(X)")
# plt.show()
#
# # Print regression performance
# print(f"Regression Test MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
#
# # --------------------------
# # 🚀 CLASSIFICATION TEST (SPIRAL DATA)
# # --------------------------
# def generate_spiral_data(samples, classes):
#     """Generates 2D spiral dataset"""
#     X = []
#     y = []
#     for i in range(classes):
#         theta = np.linspace(0, 2 * np.pi, samples)
#         r = np.linspace(0.1, 1, samples)
#         X1 = r * np.sin(theta + i * np.pi / classes)
#         X2 = r * np.cos(theta + i * np.pi / classes)
#         X.append(np.column_stack([X1, X2]))
#         y.append(np.full(samples, i))
#     return np.vstack(X), np.hstack(y)
#
# # Generate spiral data
# X_cls, y_cls = generate_spiral_data(samples=1000, classes=2)
#
# # Split into train and test sets
# X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
#
# # Initialize and train RVFL classifier
# classifier = RVFLClassifier(in_dim=2, n_nodes=50, direct_link=True)
# classifier.fit(X_train_cls, y_train_cls)
#
# # Predict on test data
# y_pred_cls = classifier.predict(X_test_cls)
#
# # Plot classification results
# plt.figure(figsize=(8, 5))
# plt.scatter(X_test_cls[:, 0], X_test_cls[:, 1], c=y_test_cls, cmap="coolwarm", label="True Labels", alpha=0.6)
# plt.scatter(X_test_cls[:, 0], X_test_cls[:, 1], c=y_pred_cls, cmap="coolwarm", marker="x", label="Predicted Labels")
# plt.legend()
# plt.title("RVFL Classification on Spiral Data")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()
#
# # Print classification performance
# print(f"Classification Test Accuracy: {accuracy_score(y_test_cls, y_pred_cls) * 100:.2f}%")
#
# # --------------------------
# # 🚀 CLASSIFICATION TEST (MAKE_MOONS DATA)
# # --------------------------
# from sklearn.datasets import make_moons
#
# # Generate Moon dataset
# X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)
#
# # Split into train and test sets
# X_train_moon, X_test_moon, y_train_moon, y_test_moon = train_test_split(X_moon, y_moon, test_size=0.2, random_state=42)
#
# # Train RVFL classifier
# classifier_moon = RVFLClassifier(in_dim=2, n_nodes=50, direct_link=True)
# classifier_moon.fit(X_train_moon, y_train_moon)
#
# # Predict
# y_pred_moon = classifier_moon.predict(X_test_moon)
#
# # Plot decision boundary for Moon dataset
# plt.figure(figsize=(8, 5))
# plt.scatter(X_test_moon[:, 0], X_test_moon[:, 1], c=y_test_moon, cmap="coolwarm", label="True Labels", alpha=0.6)
# plt.scatter(X_test_moon[:, 0], X_test_moon[:, 1], c=y_pred_moon, cmap="coolwarm", marker="x", label="Predicted Labels")
# plt.legend()
# plt.title("RVFL Classification on Make Moons Data")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()
#
# # Print classification performance
# print(f"Classification Test Accuracy (Make Moons): {accuracy_score(y_test_moon, y_pred_moon) * 100:.2f}%")
#
# # Import necessary libraries
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, accuracy_score
# from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, BaggingRegressor, BaggingClassifier
#
# # Import models from random-ml
from randomml.mlpedrvfl import MLPedRVFL
from randomml.ensemble import RVFLBaggingRegressor, RVFLBaggingClassifier, RVFLBoostingRegressor, RVFLBoostingClassifier

# --------------------------
# 🚀 REGRESSION TEST (SINE CURVE) with MLPedRVFL, Bagging, and Boosting
# --------------------------

# Generate sine wave data
X_reg = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y_reg = np.sin(X_reg).ravel()

# Split into train and test sets
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
#
# # Initialize models
# mlpedrvfl_reg = MLPedRVFL(in_dim=1, n_nodes=50, n_layers=3, aggregate="mean")
# bagging_reg = RVFLBaggingRegressor(in_dim = 1,n_estimators=10, random_state=42)
# boosting_reg = RVFLBoostingRegressor(in_dim = 1,n_estimators=10, random_state=42)
#
# # Train models
# mlpedrvfl_reg.fit(X_train_reg, y_train_reg)
# bagging_reg.fit(X_train_reg, y_train_reg)
# boosting_reg.fit(X_train_reg, y_train_reg)
#
# # Predict
# y_pred_mlpedrvfl = mlpedrvfl_reg.predict(X_test_reg)
# y_pred_bagging = bagging_reg.predict(X_test_reg)
# y_pred_boosting = boosting_reg.predict(X_test_reg)
#
# # Plot results
# plt.figure(figsize=(8, 5))
# plt.scatter(X_test_reg, y_test_reg, color="blue", label="True Sine Wave")
# plt.scatter(X_test_reg, y_pred_mlpedrvfl, color="red", label="MLPedRVFL Predictions", marker="x")
# plt.scatter(X_test_reg, y_pred_bagging, color="green", label="Bagging Predictions", marker="o")
# plt.scatter(X_test_reg, y_pred_boosting, color="purple", label="Boosting Predictions", marker="s")
# plt.plot(X_reg, np.sin(X_reg), "g--", alpha=0.6)  # Ground truth curve
# plt.legend()
# plt.title("Regression Comparison: MLPedRVFL, Bagging, and Boosting")
# plt.xlabel("X")
# plt.ylabel("sin(X)")
# plt.show()
#
# # Print performance metrics
# print(f"MLPedRVFL Regression MSE: {mean_squared_error(y_test_reg, y_pred_mlpedrvfl):.4f}")
# print(f"Bagging Regression MSE: {mean_squared_error(y_test_reg, y_pred_bagging):.4f}")
# print(f"Boosting Regression MSE: {mean_squared_error(y_test_reg, y_pred_boosting):.4f}")

# --------------------------
# 🚀 CLASSIFICATION TEST (SPIRAL DATA) with MLPedRVFL, Bagging, and Boosting
# --------------------------

def generate_spiral_data(samples, classes):
    """Generates 2D spiral dataset"""
    X = []
    y = []
    for i in range(classes):
        theta = np.linspace(0, 2 * np.pi, samples)
        r = np.linspace(0.1, 1, samples)
        X1 = r * np.sin(theta + i * np.pi / classes)
        X2 = r * np.cos(theta + i * np.pi / classes)
        X.append(np.column_stack([X1, X2]))
        y.append(np.full(samples, i))
    return np.vstack(X), np.hstack(y)

# Generate spiral data
X_cls, y_cls = generate_spiral_data(samples=100, classes=2)

# Split into train and test sets
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Initialize models
mlpedrvfl_cls = MLPedRVFL(in_dim=2, n_nodes=50, n_layers=3, aggregate="mean")
bagging_cls = RVFLBaggingClassifier(in_dim = 2,n_estimators=10, random_state=42)
boosting_cls = RVFLBoostingClassifier(in_dim = 2,n_estimators=10, random_state=42)

# Train models
mlpedrvfl_cls.fit(X_train_cls, y_train_cls)
bagging_cls.fit(X_train_cls, y_train_cls)
boosting_cls.fit(X_train_cls, y_train_cls)

# Predict
y_pred_mlpedrvfl = mlpedrvfl_cls.predict(X_test_cls)
y_pred_bagging = bagging_cls.predict(X_test_cls)
y_pred_boosting = boosting_cls.predict(X_test_cls)

# Plot classification results
plt.figure(figsize=(8, 5))
plt.scatter(X_test_cls[:, 0], X_test_cls[:, 1], c=y_test_cls, cmap="coolwarm", label="True Labels", alpha=0.6)
plt.scatter(X_test_cls[:, 0], X_test_cls[:, 1], c=y_pred_mlpedrvfl, cmap="coolwarm", marker="x", label="MLPedRVFL Predictions")
plt.legend()
plt.title("Classification: MLPedRVFL on Spiral Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Print classification performance
#print(f"MLPedRVFL Classification Accuracy: {accuracy_score(y_test_cls, y_pred_mlpedrvfl) * 100:.2f}%")
print(f"Bagging Classification Accuracy: {accuracy_score(y_test_cls, y_pred_bagging) * 100:.2f}%")
print(f"Boosting Classification Accuracy: {accuracy_score(y_test_cls, y_pred_boosting) * 100:.2f}%")


