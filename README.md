# Predicción de Rendimiento Académico con Regresión y Modelos de Regularización

## Índice

- [Índice](#índice)
- [Introducción](#introducción)
- [Métodos Utilizados](#métodos-utilizados)
- [Tecnologías](#tecnologías)
- [Descarga y Configuración](#descarga-y-configuración)
- [Requisitos Previos](#requisitos-previos)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Declaración del Problema](#declaración-del-problema)
- [Objetivo Comercial](#objetivo-comercial)
- [Preparación de Datos](#preparación-de-datos)
- [Construcción y Evaluación del Modelo](#construcción-y-evaluación-del-modelo)
- [Conclusiones](#conclusiones)
- [Regresión Ridge](#regresión-ridge)
- [Regresión Lasso](#regresión-lasso)
- [Regresión ElasticNet](#regresión-elasticnet)
- [Las Variables Más Significativas Son](#las-variables-más-significativas-son)

## Introducción

Este proyecto tiene como propósito construir un modelo predictivo para estimar el rendimiento académico de estudiantes a partir de variables sociodemográficas, académicas y de contexto. El análisis se desarrolla mediante técnicas de regresión lineal y modelos de regularización, con el fin de identificar patrones relevantes y generar predicciones útiles para la toma de decisiones educativas.

El estudio se basa en un conjunto de datos que contiene información de estudiantes, permitiendo analizar cómo diferentes factores influyen en su desempeño académico. A través de esta propuesta, se busca aportar una herramienta de apoyo para instituciones educativas, docentes y tomadores de decisiones interesados en anticipar resultados y diseñar estrategias de intervención.

### Métodos Utilizados

- Análisis exploratorio de datos.
- Limpieza y preparación de variables.
- Transformación de datos categóricos.
- División del conjunto de datos en entrenamiento y prueba.
- Escalado de características mediante StandardScaler.
- Selección de variables mediante técnicas como RFE y análisis de multicolinealidad con VIF.
- Modelado con regresión lineal y modelos de regularización: Ridge, Lasso y ElasticNet.
- Evaluación del desempeño mediante métricas como $R^2$ y RMSE.
- Análisis de residuos y validación de supuestos del modelo.

### Tecnologías

- Python
- Pandas
- NumPy
- Scikit-learn
- PyCaret
- Matplotlib
- Seaborn
- Jupyter Notebook

## Descarga y Configuración

### Requisitos Previos

Este proyecto necesita que Anaconda esté instalado en la computadora.

Para más detalles sobre la instalación, visite: https://docs.anaconda.com/anaconda/install/index.html

### Cómo Ejecutar

Puede descargar el código fuente clonando este repositorio usando Git:

1. Abra su aplicación Terminal favorita (Unix, Linux o macOS), como Terminal, Command Prompt, Consola, iTerm2, etc.
2. Clone el repositorio:

```bash
git clone <GITHUB_REPO_URL>
```

3. Abra el archivo notebook en Anaconda o ejecútelo desde Jupyter:

```bash
jupyter notebook <FILE.ipynb>
```

## Declaración del Problema

El rendimiento académico de los estudiantes puede verse afectado por múltiples factores, entre ellos el nivel educativo de los padres, el género, la preparación previa, el acceso a recursos educativos y otras variables contextuales. Identificar cuáles de estos factores tienen mayor impacto permite diseñar intervenciones más efectivas y mejorar los resultados escolares.

Sin embargo, trabajar con datos educativos implica desafíos como la presencia de variables categóricas, posibles valores atípicos, relaciones entre predictores y la necesidad de seleccionar las variables más relevantes para obtener un modelo estable y explicable.

### Objetivo Comercial

El objetivo comercial de este proyecto es apoyar a instituciones educativas y organizaciones de formación en la identificación temprana de factores asociados al bajo rendimiento académico. Esto permite mejorar la planificación de estrategias académicas, personalizar intervenciones y optimizar la asignación de recursos para aumentar la retención y el éxito estudiantil.

---

### Preparación de Datos

1. Limpieza de datos y análisis de datos faltantes.
2. Análisis y tratamiento de valores atípicos.
3. Derivación de columnas categóricas.
4. Análisis univariable.
5. Análisis bivariable.
6. Análisis multivariable.

### Construcción y Evaluación del Modelo

1. División de los datos en conjuntos de entrenamiento y prueba.
2. Escalado de características mediante StandardScaler.
3. Ingeniería y selección de características usando RFE y el Factor de Inflación de Varianza (VIF).
4. Modelado con regresión lineal usando PyCaret.
5. Implementación de modelos de regularización: Ridge, Lasso y ElasticNet.
6. Análisis de residuos.
7. Evaluación y validación del modelo.
8. Predicción sobre nuevos datos.
9. Conclusión y análisis final.

### Conclusiones

Los modelos de regresión regularizada muestran que es posible obtener predicciones razonables del rendimiento académico utilizando variables explicativas relacionadas con contexto, antecedentes y características estudiantiles. La comparación entre Ridge, Lasso y ElasticNet permite observar cómo la regularización mejora la estabilidad del modelo y reduce el riesgo de sobreajuste.

### Conclusions

A continuación se presentan los resultados obtenidos en la evaluación de los modelos de regresión:

- R2 Score para Ridge Regression: valor a completar según el resultado del notebook.
- R2 Score para Lasso Regression: valor a completar según el resultado del notebook.
- R2 Score para ElasticNet Regression: valor a completar según el resultado del notebook.

#### Ridge Regression (Segun PyCaret)

- **Optimal Lambda Value:** valor a completar según el resultado del notebook.
- **R2 Score Train:** valor a completar según el resultado del notebook.
- **R2 Test Score:** valor a completar según el resultado del notebook.
- **RMSE Test:** valor a completar según el resultado del notebook.

#### Lasso Regression (Segun PyCaret)

- **Optimal Lambda Value:** valor a completar según el resultado del notebook.
- **R2 Score Train:** valor a completar según el resultado del notebook.
- **R2 Test Score:** valor a completar según el resultado del notebook.
- **RMSE Test:** valor a completar según el resultado del notebook.

#### ElasticNet Regression (Segun PyCaret)

- **Optimal Lambda Value:** valor a completar según el resultado del notebook.
- **R2 Score Train:** valor a completar según el resultado del notebook.
- **R2 Test Score:** valor a completar según el resultado del notebook.
- **RMSE Test:** valor a completar según el resultado del notebook.

#### Las Variables Más Significativas Son:

- Variables de contexto familiar y socioeconómico.
- Variables asociadas al nivel educativo previo o preparación académica.
- Variables relacionadas con el desempeño en evaluaciones previas.
- Variables que reflejan apoyo académico y recursos disponibles.
