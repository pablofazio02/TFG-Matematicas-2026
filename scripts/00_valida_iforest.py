"""
Validación Unitaria 1: Isolation Forest
- Comparativa directa contra scikit-learn
- Reproducción de curvas de nivel (Liu et al., 2012, Fig. 2)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest as SklearnIForest
from scipy.stats import spearmanr

from src.isolation_forest import IsolationForest as CustomIForest

# 1. Crear dataset bimodal (dos centros gaussianos)
np.random.seed(42)
cluster1 = np.random.normal(loc=[-2.0, -2.0], scale=0.8, size=(150, 2))
cluster2 = np.random.normal(loc=[2.0, 2.0], scale=0.8, size=(150, 2))
X = np.vstack([cluster1, cluster2])

# 2. Grid de evaluación 2D
grid_res = 60
x_vals = np.linspace(-6, 6, grid_res)
y_vals = np.linspace(-6, 6, grid_res)
xx, yy = np.meshgrid(x_vals, y_vals)
X_grid = np.c_[xx.ravel(), yy.ravel()]

# 3. Ajuste de ambos modelos
# Modelo Propio
custom_if = CustomIForest(n_estimators=100, subsampling_size=128)
custom_if.fit(X)
scores_custom, _ = custom_if.anomaly_score(X_grid)

# Modelo Scikit-Learn (decision_function o score_samples negativo)
sklearn_if = SklearnIForest(n_estimators=100, max_samples=128)
sklearn_if.fit(X)
# En sklearn, score_samples da valores negativos donde más negativo = más anómalo
scores_sklearn = -sklearn_if.score_samples(X_grid)

# 4. Verificación numérica de consistencia
rho, _ = spearmanr(scores_custom, scores_sklearn)
print(f"==================================================")
print(f"  VALIDACIÓN ISOLATION FOREST VS SCIKIT-LEARN")
print(f"==================================================")
print(f"Correlación de Spearman en el grid: {rho:.4f}")
assert rho > 0.90, "Error: La correlación con sklearn debe ser superior a 0.90"
print(">>> TEST DE CONSISTENCIA CON SKLEARN SUPERADO <<<\n")

# 5. Generar visualización
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Contornos modelo propio
c1 = axes[0].contourf(xx, yy, scores_custom.reshape(xx.shape), levels=15, cmap="YlOrRd")
axes[0].scatter(X[:, 0], X[:, 1], c="black", s=8, alpha=0.4)
axes[0].set_title("Implementación Propia ($s(x)$)")
axes[0].set_xlabel("$X_1$")
axes[0].set_ylabel("$X_2$")
fig.colorbar(c1, ax=axes[0])

# Contornos scikit-learn
c2 = axes[1].contourf(xx, yy, scores_sklearn.reshape(xx.shape), levels=15, cmap="YlOrRd")
axes[1].scatter(X[:, 0], X[:, 1], c="black", s=8, alpha=0.4)
axes[1].set_title("Referencia: Scikit-Learn ($-\\mathrm{score\\_samples}$)")
axes[1].set_xlabel("$X_1$")
axes[1].set_ylabel("$X_2$")
fig.colorbar(c2, ax=axes[1])

plt.suptitle("Validación Unitaria: Isolation Forest sobre Distribución Bimodal", fontsize=13)
plt.tight_layout()
plt.show()