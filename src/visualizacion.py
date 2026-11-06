"""
Funciones de dibujo de los cuadernos. Etiquetas en castellano; las figuras se
guardan en PDF en la carpeta figures/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, kendalltau, rankdata, wilcoxon

from metricas import pertenencia_cola

FIGURAS = Path(__file__).resolve().parents[1] / "figures"
COLORES = {"iForest": "tab:blue", "EIF": "tab:orange", "HD": "tab:green"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.fontsize": 8})


def guardar(fig, nombre):
    """Guarda la figura en figures/<nombre>.pdf."""
    FIGURAS.mkdir(exist_ok=True)
    fig.savefig(FIGURAS / f"{nombre}.pdf", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Muestra
# ---------------------------------------------------------------------------
def elipse(mu, Sigma, nivel, num=200):
    """Elipse {x : d_M(x)^2 = chi2_{2, nivel}} de la normal N_2(mu, Sigma)."""
    r = np.sqrt(chi2.ppf(nivel, df=2))
    valores, vectores = np.linalg.eigh(Sigma)
    raiz = vectores @ np.diag(np.sqrt(valores)) @ vectores.T
    a = np.linspace(0, 2 * np.pi, num)
    return mu + r * (raiz @ np.vstack([np.cos(a), np.sin(a)])).T


def dibujar_muestra(X, mu, Sigma, i_in, i_out):
    fig, eje = plt.subplots(figsize=(5.5, 4.3))
    eje.scatter(X[:, 0], X[:, 1], s=6, c="0.6", label="Muestra")
    for nivel in (0.5, 0.9, 0.99):
        e = elipse(mu, Sigma, nivel)
        eje.plot(e[:, 0], e[:, 1], color="0.2", lw=0.8)
        eje.annotate(f"{int(nivel * 100)} %", e[25], fontsize=7, color="0.2")
    eje.scatter(*X[i_in], s=60, c="tab:blue", edgecolors="white", zorder=3, label=r"$\mathbf{x}_{\mathrm{in}}$")
    eje.scatter(*X[i_out], s=60, c="tab:red", edgecolors="white", zorder=3, label=r"$\mathbf{x}_{\mathrm{out}}$")
    eje.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    return fig


# ---------------------------------------------------------------------------
# Aislamiento de un punto en un árbol
# ---------------------------------------------------------------------------
def _recortar(poligono, n, p, izquierda):
    """Intersección de un polígono convexo con el semiplano (y - p)·n <= 0 (izquierda) o >= 0."""
    signo = 1.0 if izquierda else -1.0
    salida = []
    for i in range(len(poligono)):
        a, b = poligono[i], poligono[(i + 1) % len(poligono)]
        fa, fb = signo * float((a - p) @ n), signo * float((b - p) @ n)
        if fa <= 0:
            salida.append(a)
        if fa * fb < 0:
            salida.append(a + (b - a) * fa / (fa - fb))
    return salida


def _segmento(poligono, n, p):
    """Segmento de la recta (y - p)·n = 0 contenido en un polígono convexo."""
    puntos = []
    for i in range(len(poligono)):
        a, b = poligono[i], poligono[(i + 1) % len(poligono)]
        fa, fb = float((a - p) @ n), float((b - p) @ n)
        if fa * fb < 0:
            puntos.append(a + (b - a) * fa / (fa - fb))
    return puntos[:2] if len(puntos) >= 2 else None


def recorrido(x, T, caja):
    """
    Cortes que atraviesa x en un árbol (de iForest o del EIF) y celda final.
    Un corte de iForest (q, p) es la recta {y : y_q = p}; uno del EIF (n, p), la
    recta {y : (y - p)·n = 0}. La celda es un polígono convexo.
    """
    celda, cortes = list(caja), []
    while "q" in T or "n" in T:
        if "q" in T:
            n = np.eye(2)[T["q"]]
            p = np.full(2, T["p"])
            izquierda = x[T["q"]] < T["p"]
        else:
            n, p = T["n"], T["p"]
            izquierda = (x - p) @ n <= 0
        s = _segmento(celda, n, p)
        if s is not None:
            cortes.append(s)
        celda = _recortar(celda, n, p, izquierda)
        T = T["izq"] if izquierda else T["der"]
    return cortes, celda


def dibujar_aislamiento(sub, arboles, x_in, x_out):
    """Una fila por método (arboles = {"iForest": T, "EIF": T}): cortes que aíslan a x_in y a x_out."""
    x0, x1 = sub[:, 0].min() - 0.3, sub[:, 0].max() + 0.3
    y0, y1 = sub[:, 1].min() - 0.3, sub[:, 1].max() + 0.3
    caja = [np.array(v, float) for v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    fig, ejes = plt.subplots(len(arboles), 2, figsize=(10, 4.2 * len(arboles)), sharex=True, sharey=True,
                             squeeze=False)
    for fila, (metodo, T) in enumerate(arboles.items()):
        for col, (x, color, nombre) in enumerate(((x_in, "tab:blue", r"\mathbf{x}_{\mathrm{in}}"),
                                                  (x_out, "tab:red", r"\mathbf{x}_{\mathrm{out}}"))):
            eje = ejes[fila, col]
            cortes, celda = recorrido(x, T, caja)
            eje.scatter(sub[:, 0], sub[:, 1], s=8, facecolors="none", edgecolors="0.5")
            for a, b in cortes:
                eje.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=0.8)
            eje.add_patch(plt.Polygon(np.array(celda), closed=True, color=color, alpha=0.25))
            eje.scatter(*x, s=50, c=color, edgecolors="white", zorder=3)
            eje.set(xlim=(x0, x1), ylim=(y0, y1), title=f"{metodo}: aislamiento de ${nombre}$ ({len(cortes)} cortes)")
            if fila == len(arboles) - 1:
                eje.set_xlabel("$x_1$")
            if col == 0:
                eje.set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convergencia y puntuación de anomalía
# ---------------------------------------------------------------------------
def dibujar_convergencia(H, i_in, i_out, c_psi):
    """Media acumulada de h(x_in) y h(x_out) frente al número de árboles (H = {"iForest": matriz n x t, ...})."""
    fig, eje = plt.subplots(figsize=(6.5, 4))
    for metodo, Hm in H.items():
        t = np.arange(1, Hm.shape[1] + 1)
        for i, estilo, nombre in ((i_in, "-", r"\mathbf{x}_{\mathrm{in}}"), (i_out, "--", r"\mathbf{x}_{\mathrm{out}}")):
            eje.plot(t, np.cumsum(Hm[i]) / t, color=COLORES[metodo], ls=estilo, lw=1.3,
                     label=f"{metodo}, ${nombre}$")
    eje.axhline(c_psi, color="0.5", lw=0.8, ls=":", label=r"$c(\psi)$")
    eje.set(xscale="log", ylim=(0, None), xlabel="Número de árboles $t$ (escala logarítmica)",
            ylabel=r"$\bar h_t(\mathbf{x})$")
    eje.legend(ncol=2)
    return fig


def dibujar_puntuacion(h_media, c):
    """Curva s = 2^(-E(h)/c(psi)) para varios psi y observaciones de la muestra (psi = 256)."""
    fig, eje = plt.subplots(figsize=(6.5, 4))
    E = np.linspace(0, 30, 300)
    for psi, estilo in ((16, ":"), (64, "--"), (256, "-")):
        eje.plot(E, 2 ** (-E / c(psi)), color="0.3", ls=estilo, lw=1, label=rf"$\psi = {psi}$")
    eje.scatter(h_media, 2 ** (-h_media / c(256)), s=10, c="tab:blue", alpha=0.5, zorder=3, label="Muestra")
    eje.axhline(0.5, color="tab:red", ls="--", lw=1, label="$s = 0.5$")
    eje.set(xlabel=r"$E(h(\mathbf{x}))$", ylabel=r"$s(\mathbf{x}, \psi)$", ylim=(0, 1))
    eje.legend()
    return fig


# ---------------------------------------------------------------------------
# Comparación con la profundidad de Tukey
# ---------------------------------------------------------------------------
def dibujar_h_frente_hd(puntuaciones, hd, destacados=()):
    """
    Longitud de ruta media frente a HD(x, P_n), una columna por método y dos escalas:
      - fila superior, rangos reescalados a [0, 1]: es la escala en la que trabajan tau_b y rho_s;
        si los dos órdenes coincidieran, los puntos estarían sobre la diagonal;
      - fila inferior, valores originales con HD en escala logarítmica: muestra la forma de la
        relación; la recta es el ajuste por mínimos cuadrados de h frente a log HD (medida r_log).
    """
    n = len(hd)
    k = len(puntuaciones)
    fig, ejes = plt.subplots(2, k, figsize=(5 * k, 8.4), squeeze=False)
    r_hd = rankdata(hd) / n
    log_hd = np.log(hd)
    for j, (metodo, h) in enumerate(puntuaciones.items()):
        r_h = rankdata(h) / n
        tau = kendalltau(h, hd, variant="b").statistic
        r_log = np.corrcoef(h, log_hd)[0, 1]
        eje = ejes[0, j]
        eje.scatter(r_hd, r_h, s=6, c=COLORES[metodo], alpha=0.5)
        eje.plot([0, 1], [0, 1], color="0.4", lw=0.8, ls=":")
        eje.set(xlabel=r"Rango de $HD(\mathbf{x}, P_n)$ (escala $[0, 1]$)",
                title=rf"{metodo}: rangos, $\tau_b = {tau:.3f}$")
        eje = ejes[1, j]
        eje.scatter(hd, h, s=6, c=COLORES[metodo], alpha=0.5)
        b, a = np.polyfit(log_hd, h, 1)
        malla_hd = np.geomspace(hd.min(), hd.max(), 100)
        eje.plot(malla_hd, a + b * np.log(malla_hd), color="0.2", lw=1,
                 label=rf"$\bar h_t = {a:.2f} + {b:.2f}\,\log HD$")
        eje.set(xscale="log", xlabel=r"$HD(\mathbf{x}, P_n)$ (escala logarítmica)",
                title=rf"{metodo}: valores, $r_{{\log}} = {r_log:.3f}$")
        for i, color, nombre in destacados:
            ejes[0, j].scatter(r_hd[i], r_h[i], s=60, c=color, edgecolors="white", zorder=3, label=nombre)
            eje.scatter(hd[i], h[i], s=60, c=color, edgecolors="white", zorder=3, label=nombre)
        ejes[0, j].legend(loc="lower right")
        eje.legend(loc="lower right")
    ejes[0, 0].set_ylabel(r"Rango de $\bar h_t(\mathbf{x})$ (escala $[0, 1]$)")
    ejes[1, 0].set_ylabel(r"$\bar h_t(\mathbf{x})$")
    fig.tight_layout()
    return fig


def dibujar_mapas(X, puntuaciones):
    """La muestra coloreada por el rango de cada puntuación (más oscuro = más central)."""
    n = len(X)
    fig, ejes = plt.subplots(1, len(puntuaciones), figsize=(4.6 * len(puntuaciones), 4.2), sharey=True, squeeze=False)
    for eje, (nombre, valores) in zip(ejes[0], puntuaciones.items()):
        im = eje.scatter(X[:, 0], X[:, 1], c=rankdata(valores) / n, s=8, cmap="viridis")
        eje.set(xlabel="$x_1$", title=nombre, aspect="equal")
    ejes[0, 0].set_ylabel("$x_2$")
    fig.colorbar(im, ax=ejes[0].tolist(), shrink=0.85, label="Rango (escala $[0, 1]$)")
    return fig


def dibujar_anomalias(X, puntuaciones, hd, proporcion=0.05):
    """El 5 % menos profundo según cada método frente al 5 % menos profundo según HD(x, P_n)."""
    k = int(proporcion * len(X))
    w_hd = pertenencia_cola(hd, k)
    fig, ejes = plt.subplots(1, len(puntuaciones), figsize=(5 * len(puntuaciones), 4.3), sharey=True, squeeze=False)
    for eje, (metodo, h) in zip(ejes[0], puntuaciones.items()):
        w = pertenencia_cola(h, k)
        eje.scatter(X[:, 0], X[:, 1], s=5, c="0.8")
        solo_hd, solo_m, ambos = (w_hd > 0) & (w == 0), (w > 0) & (w_hd == 0), (w > 0) & (w_hd > 0)
        eje.scatter(*X[solo_hd].T, s=25, facecolors="none", edgecolors=COLORES["HD"], label="Solo Tukey")
        eje.scatter(*X[solo_m].T, s=25, marker="x", c=COLORES[metodo], label=f"Solo {metodo}")
        eje.scatter(*X[ambos].T, s=25, c="black", label="Ambos")
        eje.set(xlabel="$x_1$", aspect="equal",
                title=f"{metodo}: solapamiento del 5 % = {float(w @ w_hd / k):.2f}")
        eje.legend(loc="lower right")
    ejes[0, 0].set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Resultados agregados sobre las réplicas
# ---------------------------------------------------------------------------
MEDIDAS = {"tau_b": r"$\tau_b$", "rho_s": r"$\rho_S$", "r_log": r"$r_{\log}$", "xi": r"$\xi_n$",
           "sol05": "Solapamiento 5 %", "tau_centro": r"$\tau_b$ centro", "tau_periferia": r"$\tau_b$ periferia"}


def tabla_resumen(df, referencia="HD_n", variantes=("iForest", "EIF", "Oráculo HD_P")):
    """Media (desviación típica) de cada medida sobre las réplicas, por escenario y método."""
    d = df[(df["referencia"] == referencia) & df["variante"].isin(variantes)]
    filas = []
    for (esc, var), g in d.groupby(["escenario", "variante"], sort=False):
        fila = {"escenario": esc, "método": "Techo" if var.startswith("Oráculo") else var, "R": g["r"].nunique()}
        for c, nombre in MEDIDAS.items():
            fila[nombre] = f"{g[c].mean():.3f} ({g[c].std():.3f})"
        filas.append(fila)
    return pd.DataFrame(filas).set_index(["escenario", "método"])


def comparacion_pareada(df, referencia="HD_n", medidas=("tau_b", "sol05")):
    """EIF menos iForest en cada réplica: media de la diferencia, réplicas a favor del EIF y p de Wilcoxon."""
    d = df[df["referencia"] == referencia]
    filas = []
    for esc, g in d.groupby("escenario", sort=False):
        a = g[g["variante"] == "iForest"].set_index("r")
        b = g[g["variante"] == "EIF"].set_index("r")
        for c in medidas:
            dif = (b[c] - a[c]).dropna()
            filas.append({"escenario": esc, "medida": MEDIDAS[c], "diferencia media": round(dif.mean(), 4),
                          "réplicas a favor del EIF": f"{(dif > 0).sum()} / {len(dif)}",
                          "p de Wilcoxon": wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0})
    return pd.DataFrame(filas).set_index(["escenario", "medida"])


# ---------------------------------------------------------------------------
# Mapas en una malla (E3 y E5)
# ---------------------------------------------------------------------------
def malla(X, puntos=120, margen=0.5):
    """Malla regular que cubre la muestra X, como matriz de puntos (puntos^2 x 2) y ejes."""
    x = np.linspace(X[:, 0].min() - margen, X[:, 0].max() + margen, puntos)
    y = np.linspace(X[:, 1].min() - margen, X[:, 1].max() + margen, puntos)
    G = np.array(np.meshgrid(x, y)).reshape(2, -1).T
    return G, x, y


def dibujar_mapas_malla(X, x, y, mapas, filas=None):
    """Mapas de nivel de varias medidas evaluadas en la malla, con la muestra encima.
    `mapas` = {título: valores en la malla}; cada mapa se representa por rangos (claro = central)."""
    nombres = list(mapas)
    filas = filas or [nombres]
    fig, ejes = plt.subplots(len(filas), len(filas[0]), figsize=(4.2 * len(filas[0]), 3.9 * len(filas)),
                             squeeze=False, sharex=True, sharey=True)
    for i, fila in enumerate(filas):
        for j, nombre in enumerate(fila):
            eje = ejes[i, j]
            v = rankdata(mapas[nombre]) / len(mapas[nombre])
            eje.contourf(x, y, v.reshape(len(y), len(x)), levels=20, cmap="viridis")
            eje.scatter(X[:, 0], X[:, 1], s=1.5, c="white", alpha=0.5)
            eje.set(title=nombre, aspect="equal")
            if i == len(filas) - 1:
                eje.set_xlabel("$x_1$")
            if j == 0:
                eje.set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# E3: medidas frente al ángulo de giro
# ---------------------------------------------------------------------------
def dibujar_frente_a_theta(df, medidas=("tau_b", "sol05"), referencia="HD_n"):
    """Media (y banda de +-1 d.t.) de cada medida frente a theta, una curva por lambda y método."""
    d = df[df["referencia"] == referencia]
    fig, ejes = plt.subplots(1, len(medidas), figsize=(5.2 * len(medidas), 4), squeeze=False)
    estilos = {1: ":", 4: "--", 16: "-"}
    for eje, c in zip(ejes[0], medidas):
        for (lam, var), g in d[d["variante"].isin(["iForest", "EIF", "Oráculo HD_P"])].groupby(["lam", "variante"]):
            if var == "Oráculo HD_P" and lam != 16:
                continue
            r = g.groupby("theta")[c].agg(["mean", "std"])
            color = "0.5" if var.startswith("Oráculo") else COLORES[var]
            nombre = "Techo" if var.startswith("Oráculo") else rf"{var}, $\lambda = {lam}$"
            eje.plot(r.index, r["mean"], color=color, ls=estilos[lam], marker="o", ms=3, label=nombre)
        eje.set(xlabel=r"Ángulo de giro $\theta$ (grados)", ylabel=MEDIDAS[c], xticks=[0, 15, 30, 45])
    fig.tight_layout()
    fig.legend(*ejes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               frameon=False)
    return fig


# ---------------------------------------------------------------------------
# E4: parámetros
# ---------------------------------------------------------------------------
ORDEN_PSI = ["16", "32", "64", "128", "256", "512", "n"]
NOMBRE_HLIM = {"completo": r"$h_{lim} = \psi - 1$", "log2": r"$h_{lim} = \lceil \log_2 \psi \rceil$"}


def medias_parametros(df, referencia="HD_n"):
    """Media de cada medida sobre las réplicas para cada escenario, método, psi, t y h_lim."""
    d = df[(df["referencia"] == referencia) & (df["variante"] != "Oráculo HD_P")].copy()
    d["psi"] = d["psi"].astype(str)
    d["t"] = d["t"].astype(int)
    return d.groupby(["escenario", "metodo", "psi", "t", "hlim"])[list(MEDIDAS)].mean().reset_index()


def dibujar_frente_a_t(m, techos, psi="256", medida="tau_b"):
    """Medida frente al número de árboles (psi fijo), por escenario, método y h_lim."""
    escenarios = list(m["escenario"].unique())
    fig, ejes = plt.subplots(1, len(escenarios), figsize=(5.2 * len(escenarios), 4), sharey=True, squeeze=False)
    for eje, esc in zip(ejes[0], escenarios):
        s = m[(m["escenario"] == esc) & (m["psi"] == psi)]
        for (metodo, hl), g in s.groupby(["metodo", "hlim"]):
            eje.plot(g["t"], g[medida], color=COLORES[metodo], ls="-" if hl == "completo" else "--", marker="o", ms=3,
                     label=f"{metodo}, {NOMBRE_HLIM[hl]}")
        eje.axhline(techos[esc], color="0.5", lw=0.8, label="Techo")
        eje.set(xscale="log", xlabel="Número de árboles $t$ (escala logarítmica)", title=esc)
    ejes[0, 0].set_ylabel(MEDIDAS[medida])
    fig.tight_layout()
    fig.legend(*ejes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               frameon=False)
    return fig


def dibujar_frente_a_psi(m, techos, t=1000, medidas=("tau_b", "sol05")):
    """Medidas frente al tamaño de submuestra (t fijo): una fila por medida y una columna por escenario."""
    escenarios = list(m["escenario"].unique())
    fig, ejes = plt.subplots(len(medidas), len(escenarios), figsize=(5.2 * len(escenarios), 3.6 * len(medidas)),
                             sharex=True, squeeze=False)
    for i, c in enumerate(medidas):
        for j, esc in enumerate(escenarios):
            eje = ejes[i, j]
            s = m[(m["escenario"] == esc) & (m["t"] == t)]
            for (metodo, hl), g in s.groupby(["metodo", "hlim"]):
                g = g.set_index("psi").reindex([p for p in ORDEN_PSI if p in set(g["psi"])])
                posiciones = [ORDEN_PSI.index(p) for p in g.index]   # misma posición de cada psi en todos los paneles
                eje.plot(posiciones, g[c], color=COLORES[metodo], ls="-" if hl == "completo" else "--",
                         marker="o", ms=3, label=f"{metodo}, {NOMBRE_HLIM[hl]}")
            if c in techos:
                eje.axhline(techos[c][esc], color="0.5", lw=0.8, label="Techo")
            eje.set_xticks(range(len(ORDEN_PSI)), ORDEN_PSI)
            if i == 0:
                eje.set_title(esc)
            if i == len(medidas) - 1:
                eje.set_xlabel(r"Tamaño de submuestra $\psi$")
            if j == 0:
                eje.set_ylabel(MEDIDAS[c])
    fig.tight_layout()
    fig.legend(*ejes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               frameon=False)
    return fig


def tabla_mejores(m, medida="tau_b"):
    """Para cada escenario: configuración estándar y la mejor según `medida`, con todas las medidas."""
    filas = []
    for esc, s in m.groupby("escenario", sort=False):
        estandar = s[(s["psi"] == "256") & (s["t"] == 100) & (s["hlim"] == "completo")]
        for _, f in estandar.iterrows():
            filas.append({"escenario": esc, "configuración": f"{f['metodo']} estándar", **f[list(MEDIDAS)].to_dict()})
        for metodo, g in s.groupby("metodo"):
            f = g.loc[g[medida].idxmax()]
            filas.append({"escenario": esc,
                          "configuración": f"{metodo}: psi = {f['psi']}, t = {f['t']}, h_lim = {f['hlim']}",
                          **f[list(MEDIDAS)].to_dict()})
    return pd.DataFrame(filas).set_index(["escenario", "configuración"]).rename(columns=MEDIDAS)


# ---------------------------------------------------------------------------
# E5: resumen por escenario
# ---------------------------------------------------------------------------
def dibujar_barras(df, escenarios, nombres, medida="tau_b", referencias=("HD_n",)):
    """Media de una medida por escenario para iForest y el EIF (y el techo frente a HD_n)."""
    fig, eje = plt.subplots(figsize=(10, 4))
    x = np.arange(len(escenarios))
    barras = [("iForest", "HD_n", COLORES["iForest"], "iForest frente a $HD_n$"),
              ("EIF", "HD_n", COLORES["EIF"], "EIF frente a $HD_n$")]
    if "densidad" in referencias:
        barras += [("iForest", "densidad", "lightsteelblue", "iForest frente a la densidad"),
                   ("EIF", "densidad", "navajowhite", "EIF frente a la densidad")]
    ancho = 0.8 / len(barras)
    for k, (var, ref, color, etiqueta) in enumerate(barras):
        d = df[(df["variante"] == var) & (df["referencia"] == ref)].groupby("escenario")[medida].mean()
        eje.bar(x + (k - (len(barras) - 1) / 2) * ancho, [d.get(e, np.nan) for e in escenarios], ancho,
                color=color, label=etiqueta)
    techo = df[df["variante"] == "Oráculo HD_P"].groupby("escenario")[medida].mean()
    eje.scatter(x, [techo.get(e, np.nan) for e in escenarios], marker="_", s=400, c="black", zorder=3, label="Techo")
    eje.axhline(0, color="0.3", lw=0.6)
    eje.set_xticks(x, nombres, rotation=20, ha="right")
    eje.set_ylabel(MEDIDAS[medida])
    fig.tight_layout()
    fig.legend(*eje.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               frameon=False)
    return fig
