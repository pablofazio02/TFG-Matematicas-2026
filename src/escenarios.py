"""
Escenarios de simulación y su profundidad de Tukey poblacional
(Capítulo 2, sección 2.1 de la memoria).

Cada escenario sabe generar una muestra (`muestrear`) y calcular HD(x, P)
(`hd_P`) en cualquier punto:
    - en las distribuciones elípticas, X = mu + A Z con Z esféricamente simétrica,
      HD(x, P) = F_1(-d_M(x)), con F_1 la función de distribución de una componente
      de Z (la normal estándar o la t de Student con nu grados de libertad);
    - en las demás (log-normal, mezclas, anillo), HD(x, P) se aproxima por la
      profundidad respecto de una muestra de referencia independiente de
      M = 20000 observaciones, generada con una semilla fija por escenario.
"""

import numpy as np
from scipy.stats import norm, t as t_student

from profundidad import hd_muestral

M_REFERENCIA = 20000


def matriz_rotacion(grados):
    a = np.deg2rad(grados)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class Escenario:
    def __init__(self, nombre, n, d, semilla_ref=None, **parametros):
        self.nombre, self.n, self.d, self.parametros = nombre, n, d, parametros
        self.semilla_ref = 770000 + sum(map(ord, nombre)) if semilla_ref is None else semilla_ref
        self._referencia = None

    def referencia(self):
        """Muestra de referencia con la que se aproxima HD(x, P)."""
        if self._referencia is None:
            self._referencia = self.muestrear(np.random.default_rng(self.semilla_ref), M_REFERENCIA)[0]
        return self._referencia

    def hd_P(self, puntos):
        return hd_muestral(self.referencia(), puntos)

    def densidad(self, puntos):
        return None


class Eliptico(Escenario):
    """X = mu + A Z con Z normal estándar o t de Student con nu grados de libertad; Sigma = A A^T."""

    def __init__(self, nombre, n, mu, A, nu=None, **parametros):
        mu = np.asarray(mu, float)
        super().__init__(nombre, n, len(mu), **parametros)
        self.mu, self.A, self.nu = mu, np.asarray(A, float), nu
        self.Sigma = self.A @ self.A.T
        self.A_inv = np.linalg.inv(self.A)

    def muestrear(self, rng, n=None):
        n = self.n if n is None else n
        Z = rng.standard_normal((n, self.d))
        if self.nu is not None:
            Z = Z / np.sqrt(rng.chisquare(self.nu, size=n) / self.nu)[:, None]
        return self.mu + Z @ self.A.T, None

    def d_M(self, puntos):
        """Distancia de Mahalanobis a mu con la matriz Sigma verdadera."""
        return np.linalg.norm((np.asarray(puntos) - self.mu) @ self.A_inv.T, axis=1)

    def hd_P(self, puntos):
        F1 = norm.cdf if self.nu is None else (lambda x: t_student.cdf(x, self.nu))
        return F1(-self.d_M(puntos))


def normal(nombre, n, mu, Sigma, **parametros):
    """Normal N_d(mu, Sigma), generada con el factor de Cholesky de Sigma."""
    return Eliptico(nombre, n, mu, np.linalg.cholesky(np.asarray(Sigma, float)), **parametros)


def normal_rotada(nombre, n, lam, grados, **parametros):
    """N_2(0, R diag(lam, 1) R^T): el eje mayor forma un ángulo `grados` con el primer eje."""
    A = matriz_rotacion(grados) @ np.diag([np.sqrt(lam), 1.0])
    return Eliptico(nombre, n, np.zeros(2), A, lam=lam, theta=grados, **parametros)


class LogNormal(Escenario):
    """X = exp(Y), componente a componente, con Y ~ N_2(0, s^2 [[1, rho], [rho, 1]])."""

    def __init__(self, nombre, n, s, rho=0.5, **parametros):
        super().__init__(nombre, n, 2, s=s, rho=rho, **parametros)
        self.L = np.linalg.cholesky(s ** 2 * np.array([[1, rho], [rho, 1]]))

    def muestrear(self, rng, n=None):
        n = self.n if n is None else n
        return np.exp(rng.standard_normal((n, 2)) @ self.L.T), None


class MezclaNormal(Escenario):
    """w N((-delta, 0), I) + (1 - w) N((delta, 0), I)."""

    def __init__(self, nombre, n, delta, w=0.5, **parametros):
        super().__init__(nombre, n, 2, delta=delta, w=w, **parametros)
        self.delta, self.w = delta, w

    def muestrear(self, rng, n=None):
        n = self.n if n is None else n
        izquierda = rng.random(n) < self.w
        centros = np.where(izquierda[:, None], [-self.delta, 0.0], [self.delta, 0.0])
        return centros + rng.standard_normal((n, 2)), None

    def densidad(self, puntos):
        p = np.asarray(puntos)
        f = lambda c: np.exp(-0.5 * ((p - c) ** 2).sum(axis=1)) / (2 * np.pi)
        return self.w * f(np.array([-self.delta, 0])) + (1 - self.w) * f(np.array([self.delta, 0]))


class Anillo(Escenario):
    """Radio 3 con ruido radial N(0, 0.3^2) y ángulo uniforme: soporte no convexo."""

    def __init__(self, nombre, n, radio=3.0, ruido=0.3, **parametros):
        super().__init__(nombre, n, 2, radio=radio, ruido=ruido, **parametros)
        self.radio, self.ruido = radio, ruido

    def muestrear(self, rng, n=None):
        n = self.n if n is None else n
        r = self.radio + self.ruido * rng.standard_normal(n)
        a = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([r * np.cos(a), r * np.sin(a)]), None

    def densidad(self, puntos):
        r = np.linalg.norm(np.asarray(puntos), axis=1)
        return norm.pdf((r - self.radio) / self.ruido) / (self.ruido * 2 * np.pi * np.maximum(r, 1e-12))
