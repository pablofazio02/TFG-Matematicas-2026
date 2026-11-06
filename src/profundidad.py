"""
Profundidad de Tukey muestral HD(x, P_n), calculada con la biblioteca data-depth
(algoritmo exacto por defecto).

La profundidad poblacional HD(x, P) de cada escenario se calcula en escenarios.py.
"""

import numpy as np

import torch_error  # noqa: F401  (permite importar data-depth sin PyTorch)
from depth.model.DepthEucl import DepthEucl


def hd_muestral(X, puntos=None, exacta=True, **opciones):
    """HD(x, P_n) de cada fila de `puntos` (por defecto, de la propia muestra X)."""
    X = np.asarray(X, dtype=float)
    puntos = X if puntos is None else np.asarray(puntos, dtype=float)
    return DepthEucl().load_dataset(X).halfspace(puntos, exact=exacta, **opciones)
