# TFG-Matematicas-2026

Código del Trabajo Fin de Grado *Propiedades Estadísticas de Metodologías de Machine Learning*
(*Statistical Properties of Machine Learning Methodologies*).

El trabajo estudia, mediante simulación, en qué medida el orden que induce la longitud de ruta media de un
bosque de aislamiento (*Isolation Forest*, Liu, Ting y Zhou, 2012) y de su versión extendida (*Extended
Isolation Forest*, Hariri, Carrasco Kind y Brunner, 2021) reproduce el que induce la profundidad de Tukey,
y en qué situaciones deja de hacerlo.

## Instalación (Windows)

```bash
py install 3.12
py -V:3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Cuadernos

| Cuaderno | Contenido |
|---|---|
| `01_E1_correlacion_baja` | Normal bivariante con correlación baja: cómo aísla cada método las observaciones, relación entre la longitud de ruta y la profundidad de Tukey (en rangos y en escala logarítmica) y medidas de asociación sobre 100 réplicas |
| `02_E2_correlacion_alta` | Normal bivariante con correlación alta: pérdida de concordancia de iForest en la periferia y mejora del EIF |
| `03_E3_orientacion` | Efecto de alargar y girar la nube de puntos, que explica las diferencias entre iForest y EIF |
| `04_E4_parametros` | Influencia del número de árboles, el tamaño de submuestra y el límite de altura |
| `05_E5_no_elipticas` | Distribuciones no elípticas, donde la relación con la profundidad de Tukey se debilita o deja de ser creciente |

Para ejecutar un cuaderno sin repetir las simulaciones:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_E1_correlacion_baja.ipynb
```

o abrirlos en Visual Studio Code o JupyterLab, con el entorno creado como núcleo.

