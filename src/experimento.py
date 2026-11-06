"""
Ejecución de las réplicas Monte Carlo de un experimento, en paralelo
(Capítulo 2, sección 2.2 de la memoria).

Semillas. En la réplica r de un experimento con semilla base B:
    datos:   numpy.random.default_rng(B + r)
    árboles: random.seed(B + r) antes de construir CADA método, de modo que el
             resultado de un método no depende de qué otros se calculen.

Los resultados se guardan en results/<archivo>.csv a medida que terminan las
réplicas; al volver a lanzar el programa, las réplicas ya guardadas se omiten.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from metodos import variantes
from metricas import metricas_par
from profundidad import hd_muestral

RESULTADOS = Path(__file__).resolve().parents[1] / "results"


def tarea(experimento, esc, r, base, specs, con_densidad=False):
    """Una réplica: genera la muestra, calcula las referencias y compara cada método con ellas."""
    semilla = base + r
    X, _ = esc.muestrear(np.random.default_rng(semilla))
    refs = {"HD_n": hd_muestral(X), "HD_P": esc.hd_P(X)}
    if con_densidad:
        refs["densidad"] = esc.densidad(X)
    comun = {"experimento": experimento, "escenario": esc.nombre, **esc.parametros,
             "n": len(X), "r": r, "semilla": semilla}

    # Techo: concordancia de la profundidad poblacional con la muestral. Se pasa
    # log HD_P (transformación creciente que no cambia las medidas basadas en rangos)
    # para que r_log compare log HD_P con log HD_n; los ceros de la aproximación de
    # HD_P se llevan por debajo del menor valor positivo.
    hd_P = refs["HD_P"]
    minimo = hd_P[hd_P > 0].min()
    oraculo = np.log(np.where(hd_P > 0, hd_P, minimo / 2))
    filas = [{**comun, "variante": "Oráculo HD_P", "referencia": "HD_n",
              **metricas_par(oraculo, refs["HD_n"])}]
    for nombre, spec in specs.items():
        random.seed(semilla)
        resultados = variantes(X, spec)
        for meta, h_media in resultados:
            variante = nombre
            if len(resultados) > 1:
                variante = f"{nombre} | t={meta['t']} | hlim={meta['hlim']}"
            for nombre_ref, prof in refs.items():
                filas.append({**comun, "variante": variante, **meta, "referencia": nombre_ref,
                              **metricas_par(h_media, prof)})
    return filas


def ejecutar(experimento, escenarios, specs, R, base, archivo=None, n_jobs=-1, con_densidad=False):
    """Lanza las réplicas pendientes y añade sus filas a results/<archivo>.csv."""
    RESULTADOS.mkdir(exist_ok=True)
    ruta = RESULTADOS / f"{archivo or experimento}.csv"
    hechas = set()
    if ruta.exists():
        hechas = set(map(tuple, pd.read_csv(ruta, usecols=["escenario", "r"]).drop_duplicates().to_numpy()))
    tareas = [(esc, r) for esc in escenarios for r in range(R) if (esc.nombre, r) not in hechas]

    generador = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(tarea)(experimento, esc, r, base, specs, con_densidad) for esc, r in tareas)
    for filas in generador:
        df = pd.DataFrame(filas)
        if ruta.exists():
            columnas = pd.read_csv(ruta, nrows=0).columns
            df = df.reindex(columns=columnas.union(df.columns, sort=False))
            if list(df.columns) != list(columnas):  # aparecen columnas nuevas: se reescribe el fichero
                pd.concat([pd.read_csv(ruta), df]).to_csv(ruta, index=False)
                continue
        df.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)
    return len(tareas)
