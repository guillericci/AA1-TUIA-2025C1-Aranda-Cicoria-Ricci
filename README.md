# Trabajo Práctico N°1 - Modelo predictivo de tarifas de Uber
## AA1-TUIA-2025C1-Aranda-Cicoria-Ricci
Este repositorio contiene el desarrollo del primer trabajo práctico de la materia **Aprendizaje Automático 1** de la **Tecnicatura en Inteligencia Artificial**. El objetivo es construir un modelo predictivo para estimar tarifas de viajes en Uber, utilizando técnicas de regresión lineal, regularización, y optimización de hiperparámetros, empleando la biblioteca `scikit-learn`.

---

### Estructura del notebook

El notebook principal se encuentra en el archivo `TP-regresion-AA1.ipynb` y está organizado de la siguiente manera:

1. **Contexto**  
   Breve introducción al problema y al dataset.

2. **Variables del conjunto de datos**  
   Descripción de las variables incluidas en el dataset `uber_fares.csv`.

3. **Carga del Dataset**  
   Lectura del archivo y primeros análisis.

4. **Análisis Exploratorio**  
   - Limpieza y filtrado de datos.  
   - Selección de variables predictoras.  
   - Visualización de datos (diagramas de dispersión, histogramas, boxplots).  
   - Análisis de correlaciones.  

5. **Preprocesamiento**  
   - Imputación de valores faltantes.  
   - Estandarización y escalado.  
   - Creación de nuevas variables (día, hora, feriado, distancia, etc.).  

6. **Modelados**  
   - **Regresión Lineal Múltiple.**  
   - **Regularización:** modelos de Ridge, Lasso y ElasticNet.  
   - **Gradiente descendente.**  

7. **Comparación de modelos**  
   - Evaluación con métricas como RMSE, R².  
   - Comparación visual de los resultados.  
   - Análisis de coeficientes.

8. **Conclusión**  
   Reflexión sobre los resultados obtenidos, desempeño de los modelos y decisiones tomadas a lo largo del trabajo.

---

### Dataset

El dataset utilizado contiene información de viajes de Uber en la ciudad de Nueva York. Las variables más relevantes utilizadas para la predicción de la tarifa fueron:

- Coordenadas geográficas (pickup/dropoff)
- Distancia estimada
- Día y hora del viaje
- Feriados (utilizando librerías como `holidays`)
- Componentes trigonométricas para mes y hora

---

### Modelos implementados

- **Regresión Lineal Múltiple (`LinearRegression`)**
- **Regresión con Gradiente Descendente**
- **Regularización:**
  - `LassoCV`
  - `RidgeCV`
  - `ElasticNetCV`

Cada uno fue ajustado, validado y evaluado con técnicas como **validación cruzada** y **búsqueda de hiperparámetros**. Se analizaron los coeficientes de los modelos y su evolución respecto a los valores de regularización (alpha).

---

### Métricas utilizadas

- **RMSE (Root Mean Squared Error)**
- **R² Score (Coeficiente de determinación)**

Estas métricas fueron calculadas tanto para el conjunto de entrenamiento como para el de prueba, con el fin de evaluar el fitting del modelo y detectar posibles problemas de sobreajuste.

---

### Requisitos para reproducir

Este proyecto fue desarrollado con Python 3.12 y las siguientes bibliotecas principales:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `holidays`
