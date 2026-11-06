"""
E5: distribuciones no elípticas, n = 1000.
    Colas pesadas y asimetría (R = 100):
        t de Student elíptica con nu en {1, 3, 10} y matriz de dispersión [[1, 0.5], [0.5, 1]];
        log-normal exp(Y), Y ~ N_2(0, s^2 [[1, 0.5], [0.5, 1]]), con s en {0.5, 1}.
    Multimodalidad y soporte no convexo (R = 50):
        0.5 N((-delta, 0), I) + 0.5 N((delta, 0), I), con delta en {0, 1.5, 3};
        0.8 N((-3, 0), I) + 0.2 N((3, 0), I);
        anillo de radio 3 con ruido radial N(0, 0.3^2).
En los modelos no elípticos, HD(x, P) se aproxima con una muestra de referencia de
20000 observaciones. Se calcula también la densidad verdadera de las mezclas y del anillo.
"""
import numpy as np

from comun import lanzar
from escenarios import Eliptico, LogNormal, MezclaNormal, Anillo

DISPERSION = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
COLAS = ([Eliptico(f"E5 t nu={nu}", 1000, [0.0, 0.0], DISPERSION, nu=nu, familia="t", nu_t=nu) for nu in (1, 3, 10)]
         + [LogNormal(f"E5 lognormal s={s}", 1000, s, familia="lognormal", semilla_ref=ref)
            for s, ref in ((0.5, 771479), (1.0, 771475))])
MULTIMODALES = ([MezclaNormal(f"E5 mezcla delta={dl}", 1000, dl, familia="mezcla") for dl in (0.0, 1.5, 3.0)]
                + [MezclaNormal("E5 mezcla 0.8/0.2 delta=3", 1000, 3.0, w=0.8, familia="mezcla desigual"),
                   Anillo("E5 anillo", 1000, familia="anillo")])

if __name__ == "__main__":
    lanzar("E5", COLAS, R_defecto=100, base=4000)                         # semilla: 4000 + r
    lanzar("E5", MULTIMODALES, R_defecto=50, base=5000, con_densidad=True)  # semilla: 5000 + r
