"""
Módulo: halfspace_depth.py
Implementación del cálculo exacto (R^2) y aproximado por proyecciones (R^d)
de la Profundidad de Semiespacios (Halfspace Depth).

Referencias:
- Rousseeuw, P. J., & Ruts, I. (1996). "Algorithm AS 307: Bivariate Location Depth". 
  Applied Statistics (JRSS-C), 45(4), 516-526.
- Dyckerhoff, R. (2004). "Data depths satisfying the projection property". 
  Allgemeines Statistisches Archiv, 88(2), 163-190.
- Cuesta-Albertos, J. A., & Nieto-Reyes, A. (2008). "The random Tukey depth". 
  Computational Statistics & Data Analysis, 52(11), 4979-4988.
"""

__author__ = "Pablo Fazio Arrabal"
__institution__ = "Universidad de Málaga (UMA)"
__degree__ = "Doble Grado en Matemáticas e Ingeniería Informática"

import numpy as np

def halfspace_depth_2d_exact(theta, data):
    """
    Calcula la profundidad de Tukey exacta para un punto 'theta' en el plano (d=2)
    con respecto a la muestra 'data'.
    
    Basado en el algoritmo de Rousseeuw y Ruts (1996) con complejidad O(n log(n)).
    El algoritmo traslada el origen a theta, ordena los ángulos polares y usa 
    una ventana deslizante para encontrar el semiespacio con menor masa empírica.
    
    Parámetros:
    -----------
    theta : numpy.ndarray
        Vector de dimensión dos (el punto a evaluar).
    data : numpy.ndarray
        Matriz de tamaño (N, 2) con las observaciones.
        
    Devuelve:
    --------
    float
        Valor exacto de la profundidad de Tukey HD(theta, data) en [0, 1].
    """

    n = len(data)

    if n == 0:
        return 0.0
    
    # Traslación respecto a theta y conteo de puntos coincidentes
    diff = data - theta
    norms = np.linalg.norm(diff, axis=1)
    
    coincidentes = np.sum(norms == 0)
    non_zero_diff = diff[norms > 0]
    
    if len(non_zero_diff) == 0:
        return 1.0

    # Ángulos polares en (-pi, pi] y ordenación
    angles = np.sort(np.arctan2(non_zero_diff[:, 1], non_zero_diff[:, 0]))
    m = len(angles)
    
    extended_angles = np.concatenate([angles, angles + 2 * np.pi])
    
    min_count = m
    right_idx = 0
    
    for left_idx in range(m):
        target = angles[left_idx] + np.pi
        
        while right_idx < 2 * m and extended_angles[right_idx] <= target + 1e-7:
            right_idx += 1
            
        count = right_idx - left_idx
        
        min_dir = min(count, m - count + 1)
        if min_dir < min_count:
            min_count = min_dir
            
    return (min_count + coincidentes) / n


def halfspace_depth_aprox(X_eval, data, n_projections=1000):
    """
    Aproximación estocástica de la profundidad de Tukey en R^d (d >= 2).
    
    Basado en el principio de proyecciones unidimensionales sobre la hiperesfera 
    unitaria S^{d-1}:
    
        HD(x, P) = min_{j=1,...,m} HD(<x, u_j>, P_{<X, u_j>})
        
    donde u_j son vectores directores muestreados uniformemente sobre S^{d-1}.
    
    Parámetros:
    -----------
    X_eval : array_like de dimensión (M, d) o (d,)
        Puntos a evaluar.
    data : array_like de dimensión (N, d)
        Muestra de datos de referencia.
    n_projections : int, opcional (por defecto 1000)
        Número de direcciones aleatorias uniformes sobre la esfera S^{d-1}.
        
    Devuelve:
    --------
    numpy.ndarray
        Vector de longitud M con las profundidades de Tukey estimadas en [0, 1].
    """

    X_eval = np.atleast_2d(X_eval)
    data = np.asarray(data, dtype=np.float64)
    n_samples, d = data.shape
    
    # Generación de vectores directores uniformes en S^{d-1}
    u = np.random.randn(n_projections, d)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    
    # Proyecciones unidimensionales
    proj_data = data @ u.T      # Matriz (N, m)
    proj_eval = X_eval @ u.T    # Matriz (M, m)
    
    depths = np.zeros(len(X_eval), dtype=np.float64)
    for i in range(len(X_eval)):
        p_left = np.sum(proj_data <= proj_eval[i, :], axis=0) / n_samples
        p_right = np.sum(proj_data >= proj_eval[i, :], axis=0) / n_samples
        depths[i] = np.min(np.minimum(p_left, p_right))
        
    return depths