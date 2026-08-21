"""
Módulo: metrics.py
Métricas de evaluación estadística y rendimiento de detección de anomalías.

Referencias:
- Liu et al. (2012), Appendix E (Algorithm 4).
- Hand, D. J., & Till, R. J. (2001). "A simple generalisation of the area 
  under the ROC curve". Machine Learning, 45(2), 171-186.
"""

__author__ = "Pablo Fazio Arrabal"
__institution__ = "Universidad de Málaga (UMA)"
__degree__ = "Doble Grado en Matemáticas e Ingeniería Informática"

import numpy as np

def compute_auc(y_true, scores):
    """
    Calcula el Área Bajo la Curva ROC (AUC) siguiendo el Algoritmo 4 de Liu et al. (2012).
    
    Parámetros:
    -----------
    y_true : array_like
        Etiquetas binarias (1 = anomalía, 0 = punto normal).
    scores : array_like
        Puntuaciones de anomalía calculadas por el modelo.
        
    Devuelve:
    --------
    float
        Valor de AUC en el intervalo [0, 1].
    """
    
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    
    n_a = int(np.sum(y_true == 1))  # Número de anomalías
    n_n = int(np.sum(y_true == 0))  # Número de puntos normales
    
    if n_a == 0 or n_n == 0:
        return 0.5
    
    # 1. Ordenar índices según scores de anomalía en orden descendente
    sorted_indices = np.argsort(-scores)
    sorted_labels = y_true[sorted_indices]
    
    # 2. Rangos de las anomalías (1-indexed en orden descendente: rank 1 = mayor score)
    # Para Hand & Till / Wilcoxon-Mann-Whitney:
    # S = suma de rangos de las instancias positivas en orden ascendente de anomalía
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    
    sum_ranks_anomalies = np.sum(ranks[y_true == 1])
    
    auc = (sum_ranks_anomalies - (n_a * (n_a + 1.0)) / 2.0) / (n_a * n_n)
    return float(auc)


def spearman_correlation(x, y):
    """
    Calcula el coeficiente de correlación de rangos de Spearman (rho)
    entre dos variables continuas.
    
    rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
    
    Parámetros:
    -----------
    x, y : array_like
        Vectores numéricos a correlacionar.
        
    Devuelve:
    --------
    float
        Coeficiente rho en [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    
    if n < 2:
        return 0.0
    
    rank_x = np.argsort(np.argsort(x)).astype(float) + 1.0
    rank_y = np.argsort(np.argsort(y)).astype(float) + 1.0
    
    d_sq = np.sum((rank_x - rank_y) ** 2)
    rho = 1.0 - (6.0 * d_sq) / (n * (n**2 - 1.0))
    return float(rho)