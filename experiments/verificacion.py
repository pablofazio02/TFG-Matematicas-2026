"""
Verificación de la implementación (Capítulo 2, sección 1.3 de la memoria). Guarda
results/verificacion.csv con una fila por comprobación.

    1. Evaluación vectorizada == recursiva (iForest y EIF; varios h_lim).
    2. Estructura de un iTree con puntos distintos (psi hojas, psi - 1 nodos internos).
    3. EIF con nivel de extensión 0 frente a iForest (Hariri et al. afirman que coinciden):
       medias y prueba de Kolmogorov-Smirnov para cuatro puntos fijos, 3000 árboles por método.
    4. Invarianza de iForest: con la misma semilla, una transformación
       afín coordenada a coordenada con a_q > 0 da exactamente las mismas longitudes de
       ruta; una rotación no.
Uso: python experiments/verificacion.py
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isolation_forest import iforest, longitudes, path_length, aplanar
from eif import eif, longitudes_eif, path_length_eif
from escenarios import normal

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    filas = []
    esc = normal("E2", 1000, [2.0, -1.0], [[2.0, 1.2], [1.2, 1.0]])
    X, _ = esc.muestrear(np.random.default_rng(2000))

    # 1. Vectorizado frente a recursivo
    random.seed(1)
    B = iforest(X, t=50, psi=256)
    random.seed(1)
    E = eif(X, t=50, psi=256)
    for hl in (3, 8, 255):
        d_if = np.abs(longitudes(X, B, hl) - np.array([[path_length(x, T, hl) for T in B] for x in X])).max()
        d_eif = np.abs(longitudes_eif(X, E, hl) - np.array([[path_length_eif(x, T, hl) for T in E] for x in X])).max()
        filas += [{"comprobacion": "vectorizado_vs_recursivo", "metodo": "iForest", "parametro": f"hlim={hl}",
                   "valor": d_if},
                  {"comprobacion": "vectorizado_vs_recursivo", "metodo": "EIF", "parametro": f"hlim={hl}",
                   "valor": d_eif}]

    # 2. Estructura
    hojas = [(aplanar(T)["izq"] < 0).sum() for T in B]
    internos = [(aplanar(T)["izq"] >= 0).sum() for T in B]
    filas += [{"comprobacion": "estructura", "metodo": "iForest", "parametro": "hojas (min, max)",
               "valor": f"{min(hojas)}, {max(hojas)}"},
              {"comprobacion": "estructura", "metodo": "iForest", "parametro": "nodos internos (min, max)",
               "valor": f"{min(internos)}, {max(internos)}"}]

    # 3. Extensión 0 frente a iForest
    puntos = np.array([[2.0, -1.0], [4.5, 0.5], [2.0, 0.5], [-1.0, -3.0]])
    random.seed(11)
    Hi = longitudes(puntos, iforest(X, t=3000, psi=256), 255)
    random.seed(12)
    H0 = longitudes_eif(puntos, eif(X, t=3000, psi=256, nivel=0), 255)
    for j, p in enumerate(puntos):
        filas += [{"comprobacion": "eif_k0", "metodo": "iForest", "parametro": f"x=({p[0]:g}, {p[1]:g}) media", "valor": Hi[j].mean()},
                  {"comprobacion": "eif_k0", "metodo": "EIF k=0", "parametro": f"x=({p[0]:g}, {p[1]:g}) media", "valor": H0[j].mean()},
                  {"comprobacion": "eif_k0", "metodo": "KS", "parametro": f"x=({p[0]:g}, {p[1]:g}) p-valor",
                   "valor": ks_2samp(Hi[j], H0[j]).pvalue}]

    # 4. Invarianza de iForest (acoplamiento con la misma semilla)
    g = lambda Y: Y * np.array([3.0, 0.5]) + np.array([-7.0, 2.0])
    a = np.deg2rad(30)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    random.seed(21)
    h0 = longitudes(X, iforest(X, t=200, psi=256), 255).mean(axis=1)
    random.seed(21)
    h1 = longitudes(g(X), iforest(g(X), t=200, psi=256), 255).mean(axis=1)
    random.seed(21)
    h2 = longitudes(X @ Rm.T, iforest(X @ Rm.T, t=200, psi=256), 255).mean(axis=1)
    filas += [{"comprobacion": "invarianza_iforest", "metodo": "iForest", "parametro": "afín por coordenadas: max |dif|",
               "valor": np.abs(h0 - h1).max()},
              {"comprobacion": "invarianza_iforest", "metodo": "iForest", "parametro": "rotación 30 grados: max |dif|",
               "valor": np.abs(h0 - h2).max()},
              {"comprobacion": "invarianza_iforest", "metodo": "iForest", "parametro": "rotación 30 grados: tau_b",
               "valor": kendalltau(h0, h2).statistic}]

    RES.mkdir(exist_ok=True)
    pd.DataFrame(filas).to_csv(RES / "verificacion.csv", index=False)
    print(pd.DataFrame(filas).to_string())


if __name__ == "__main__":
    main()
