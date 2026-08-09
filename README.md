# Predicción del rendimiento académico con aprendizaje supervisado y regresión

## Descripción general

Este proyecto desarrolla un sistema de aprendizaje supervisado para predecir el rendimiento académico de un estudiante a partir de variables contextuales y académicas. El objetivo principal es estimar la nota esperada en una materia, en este caso la variable objetivo es la columna `math score` del dataset `StudentsPerformance.csv`.

El problema se trata como un problema de regresión, porque la salida no es una clase discreta, sino un valor numérico continuo que representa una puntuación en una escala de 0 a 100. La idea central es aprender una función que relacione las características del estudiante con su rendimiento escolar.

La solución combina:

- limpieza y preparación del dataset,
- ingeniería de variables,
- validación cruzada,
- comparación de escenarios de predicción,
- evaluación del modelo con métricas de regresión,
- análisis de importancia de variables y equidad de errores por grupos.

---

## Objetivo del proyecto

El proyecto busca responder preguntas como:

- ¿Qué variables influyen más en el rendimiento matemático de los estudiantes?
- ¿Qué tan bien se puede predecir la nota final a partir de variables socioeconómicas y académicas?
- ¿El nivel educativo de los padres, la preparación previa o el tipo de almuerzo tienen impacto significativo?
- ¿La lectura y la escritura ayudan a mejorar la predicción del desempeño matemático?

Este tipo de análisis es útil para instituciones educativas, docentes y analistas que buscan detectar estudiantes en riesgo, personalizar apoyo académico y tomar decisiones basadas en evidencia.

---

## Tipo de aprendizaje que se aplica

El proyecto usa aprendizaje supervisado, porque se entrena con datos etiquetados: cada fila del dataset contiene las características del estudiante y la puntuación real obtenida.

En términos simples:

- Entrada: información del estudiante
- Etiqueta/objetivo: `math score`
- Tarea: aprender una relación entre las variables de entrada y la nota esperada

Como la variable objetivo es continua, la tarea es de regresión.

La regresión se aplica de la siguiente forma:

1. Se toma el conjunto de datos con atributos del estudiante.
2. Se separan las variables predictoras y la variable objetivo.
3. Se construye un modelo que aprende a mapear las características del estudiante a una puntuación numérica.
4. Se compara la predicción del modelo con la nota real usando métricas como RMSE, MAE y R².

En el pipeline actual del repositorio, el modelo seleccionado para la publicación final es `RandomForestRegressor`, ya que ofrece mejor desempeño que un modelo base y resultados más estables en validación.

---

## Dataset: StudentsPerformance.csv

El dataset principal del proyecto se encuentra en:

- [dataset/StudentsPerformance.csv](dataset/StudentsPerformance.csv)

Este archivo contiene 1,000 observaciones y las siguientes columnas principales:

- `gender`: género del estudiante
- `race/ethnicity`: grupo étnico
- `parental level of education`: nivel educativo de los padres
- `lunch`: tipo de almuerzo (estándar o gratis/reducido)
- `test preparation course`: preparación previa para la prueba
- `math score`: puntaje en matemáticas (variable objetivo)
- `reading score`: puntaje en lectura
- `writing score`: puntaje en escritura

Además, el script de generación crea una columna auxiliar:

- `test_prep_completed`: indica si el estudiante completó el curso de preparación.

### Cómo se utiliza el dataset

El flujo del proyecto usa el dataset en dos escenarios:

#### Escenario A

Se usa solo información de contexto del estudiante:

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`

Este escenario evalúa si es posible predecir el rendimiento usando solo variables sociodemográficas y contextuales.

#### Escenario B

Se agregan también dos variables académicas previas:

- `reading score`
- `writing score`

Este escenario representa una situación más realista y generalmente más predictiva, porque incorpora indicadores del desempeño académico anterior.

El código evidencia que el mejor desempeño se obtiene en el escenario B, lo que confirma que el rendimiento pasado en lectura y escritura aporta mucha más información que solo el contexto social o demográfico.

---

## Proceso de aprendizaje aplicado

El flujo del proyecto es el siguiente:

1. Carga del dataset desde `StudentsPerformance.csv`.
2. Limpieza de cadenas y normalización básica de texto.
3. Definición de la variable objetivo: `math score`.
4. Selección de dos conjuntos de características:
   - escenario A: variables contextuales
   - escenario B: contexto + lectura + escritura
5. División en entrenamiento y prueba con una semilla fija (`random_state=42`).
6. Preparación del preprocesamiento:
   - codificación one-hot para variables categóricas,
   - escalado para columnas numéricas cuando aplica.
7. Entrenamiento del modelo de regresión.
8. Evaluación con validación cruzada y métricas de regresión.
9. Diagnóstico de importancia de variables.
10. Evaluación de equidad por grupos.
11. Generación de reportes y artefactos (gráficas, métricas y modelos).

---

## Modelo de regresión utilizado

Aunque el proyecto explora varios enfoques, la versión final de publicación usa un modelo de regresión basado en árboles:

- `RandomForestRegressor`

Este tipo de modelo es útil para tareas de regresión porque:

- maneja bien relaciones no lineales,
- funciona con variables categóricas tras una codificación apropiada,
- no requiere que las relaciones sean lineales,
- permite analizar la importancia de variables.

En el código actual, se entrenan dos modelos base para comparación:

- `DummyRegressor` (baseline)
- `RandomForestRegressor`

La comparación permite verificar si el modelo mejora significativamente respecto a una predicción trivial basada en la media.

---

## Métricas de evaluación

La evaluación se realiza con métricas estándar de regresión:

- RMSE (Root Mean Squared Error): mide el error promedio cuadrático. Cuanto más bajo, mejor.
- MAE (Mean Absolute Error): error absoluto promedio.
- R² (coeficiente de determinación): indica qué proporción de la variabilidad se explica por el modelo.

Los resultados finales del repositorio muestran que el escenario B es el mejor:

- RMSE (holdout): 5.9972
- MAE (holdout): 4.6199
- R² (holdout): 0.8522

Esto significa que el modelo explica alrededor del 85.2% de la variabilidad del rendimiento matemático en el conjunto de prueba.

Por comparación, el escenario A tiene un rendimiento mucho menor, lo que refuerza la idea de que el historial académico previo (lectura y escritura) es clave para predecir matemáticas.

---

## Importancia de variables

El proyecto genera análisis de importancia de variables para interpretar el modelo. Los resultados más relevantes indican que los factores más influyentes suelen ser:

- `reading score`
- `writing score`
- variables de género,
- grupo étnico,
- nivel de preparación previa para pruebas,
- tipo de almuerzo,
- nivel educativo de los padres.

Esto es muy útil porque no solo se predice, sino que también se entiende “por qué” una persona puede tener un rendimiento bajo o alto.

---

## Análisis de equidad y grupos

El proyecto no se limita a medir error promedio; también analiza errores por grupo para detectar diferencias en el desempeño del modelo:

- por género,
- por grupo étnico,
- por tipo de almuerzo.

Esto ayuda a identificar si el modelo tiene un comportamiento desigual entre distintos grupos de estudiantes. Es especialmente importante en contextos educativos, donde la equidad es tan relevante como la precisión.

---

## Estructura del repositorio

- [README.md](README.md): documentación del proyecto.
- [build_publication_notebook.py](build_publication_notebook.py): script principal para cargar datos, entrenar, evaluar y generar artefactos.
- [metadata_experimento.json](metadata_experimento.json): métricas y metadatos del experimento.
- [dataset/StudentsPerformance.csv](dataset/StudentsPerformance.csv): dataset principal.
- [models/](models/): modelos entrenados en formato `.joblib`.
- [plots/](plots/): gráficos e indicadores generados por el análisis.
- [jupyter/](jupyter/): notebook exploratorio del proyecto.

---

## Cómo ejecutar este proyecto

### Requisitos

- Python 3.10+
- pip
- Bibliotecas: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, shap, pdpbox

### Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si el proyecto no tiene archivo `requirements.txt`, se pueden instalar las librerías necesarias de forma directa:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib shap pdpbox
```

### Ejecución del experimento

```bash
python build_publication_notebook.py
```

Este comando:

- carga el CSV,
- prepara los escenarios de entrenamiento,
- entrena el modelo,
- genera gráficos en la carpeta `plots`,
- guarda los modelos en `models`,
- escribe el archivo `metadata_experimento.json`.

---

## Ejemplo de uso del modelo

El script incluye una utilidad para hacer predicciones sobre un estudiante nuevo. La idea es entregar sus características y obtener la nota estimada en matemáticas.

Ejemplo conceptual:

```python
from build_publication_notebook import predecir_estudiante

pred = predecir_estudiante(
    model,
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_prep_course="completed",
    reading_score=78,
    writing_score=80,
)

print(pred)
```

Esto devuelve un valor numérico estimado del puntaje en matemáticas.

---

## Conclusiones

El proyecto demuestra que se puede predecir con buena precisión el rendimiento académico de los estudiantes a partir de variables observables y contexto escolar. El análisis confirma que la información académica previa del estudiante, especialmente las notas en lectura y escritura, aporta un poder predictivo muy alto.

En el contexto del repositorio actual, el modelo final logra un rendimiento sólido con un R² de aproximadamente 0.852 en el conjunto de prueba, lo que indica que la predicción es útil para decisiones de apoyo educativo y monitoreo de desempeño.

La combinación de predicción, interpretación y análisis de equidad hace que este proyecto sea útil tanto para investigación como para aplicaciones reales en educación.

---

## Referencias de trabajo

Este proyecto integra técnicas de:

- aprendizaje automático supervisado,
- regresión para variables continuas,
- preprocesamiento tabular,
- codificación de variables categóricas,
- validación cruzada y evaluación de modelos,
- análisis explicativo de variables.
