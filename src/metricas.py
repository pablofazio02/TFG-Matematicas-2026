"""
Medidas de asociación entre la longitud de ruta media y una profundidad
(Capítulo 2, sección 3 de la memoria).

Las dos variables se pasan orientadas como profundidades (mayor valor = observación
más central), de modo que todas las medidas son positivas cuando los órdenes concuerdan.
"""

import numpy as np
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr


def pertenencia_cola(profundidad, k):
    """
    Probabilidad de que cada observación esté entre las k menos profundas cuando
    los empates en el punto de corte se deshacen al azar: 1 para las que quedan
    estrictamente por debajo del corte y r/m para las m empatadas en el corte, de
    las que faltan r por elegir.
    """
    profundidad = np.asarray(profundidad, dtype=float)
    corte = np.sort(profundidad)[k - 1]
    debajo = profundidad < corte
    empatadas = profundidad == corte
    w = debajo.astype(float)
    w[empatadas] = (k - debajo.sum()) / empatadas.sum()
    return w


def solapamiento_cola(prof_a, prof_b, proporcion=0.05):
    """Proporción esperada de observaciones comunes entre el 100·proporcion % menos
    profundo según a y según b, con los empates deshechos al azar."""
    k = int(proporcion * len(prof_a))
    return float(pertenencia_cola(prof_a, k) @ pertenencia_cola(prof_b, k) / k)


def tau_en(mascara, a, b):
    """tau_b de Kendall restringida a las observaciones de `mascara`."""
    return kendalltau(a[mascara], b[mascara], variant="b").statistic


def xi_chatterjee(x, y, semilla=0):
    """
    Coeficiente xi_n(x, y) de Chatterjee (2021): mide en qué grado y es función de x,
    sea o no monótona. Se ordenan los pares según x (los empates en x se deshacen al
    azar, con una semilla fija) y, con r_i = #{j : y_j <= y_(i)} y l_i = #{j : y_j >= y_(i)},
        xi_n = 1 - n sum_i |r_(i+1) - r_i| / (2 sum_i l_i (n - l_i)).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    orden = np.lexsort((np.random.default_rng(semilla).random(n), x))
    ys = y[orden]
    r = rankdata(ys, method="max")
    l = n - rankdata(ys, method="min") + 1
    return float(1 - n * np.abs(np.diff(r)).sum() / (2 * (l * (n - l)).sum()))


def pearson_log(h, prof):
    """Correlación de Pearson entre h y log(prof); NaN si prof no es positiva."""
    prof = np.asarray(prof, dtype=float)
    if np.any(prof <= 0):
        return np.nan
    return pearsonr(h, np.log(prof)).statistic


def metricas_par(h, prof):
    """
    Medidas de la memoria entre la longitud de ruta media h y la profundidad prof:
        tau_b          tau_b de Kendall (corregida por empates): concordancia de los órdenes
        rho_s          rho de Spearman: correlación de Pearson entre los rangos
        r_log          correlación de Pearson entre h y log(prof): ajuste lineal en escala logarítmica
        xi             xi_n(prof, h) de Chatterjee: dependencia funcional de h respecto de prof
        sol05          solapamiento esperado del 5 % menos profundo
        tau_centro     tau_b en el 50 % más profundo según prof
        tau_periferia  tau_b en el 20 % menos profundo según prof
    """
    h, prof = np.asarray(h, dtype=float), np.asarray(prof, dtype=float)
    return {"tau_b": kendalltau(h, prof, variant="b").statistic,
            "rho_s": spearmanr(h, prof).statistic,
            "r_log": pearson_log(h, prof),
            "xi": xi_chatterjee(prof, h),
            "sol05": solapamiento_cola(h, prof, 0.05),
            "tau_centro": tau_en(prof >= np.median(prof), h, prof),
            "tau_periferia": tau_en(prof <= np.quantile(prof, 0.20), h, prof)}
