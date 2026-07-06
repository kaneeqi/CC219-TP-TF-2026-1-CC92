# Asistente Analítico NLP para E-commerce (Fonazo)

### Objetivo del trabajo
Desarrollar un Asistente Analítico basado en Procesamiento de Lenguaje Natural (NLP) capaz de automatizar la categorización de mensajes no estructurados de clientes. El sistema clasificará la intención del usuario (multiclase), el sentimiento del texto (polaridad) y predecirá la necesidad de escalamiento urgente a un asesor humano.

### Nombres de los alumnos participantes
* Valdivia Guzman, Jose Agustin (U201822153)
* Bravo Lévano, Eduardo Fernando (U20231E122)
* Cunyas Ramos, Iván Rubén (U202316818)

### Breve descripción del dataset
El proyecto utiliza un enfoque de dataset híbrido compuesto por dos fuentes:
1. **Dataset de Sentimientos (Amazon Reviews ES):** ~200,000 reseñas de e-commerce mapeadas a 2 clases de sentimiento (Positivo, Negativo). *(Ver informe adjunto para más detalles).*
2. **Dataset de Intenciones (Fonazo):** Corpus de 96 registros originales, ampliado a 296 mediante aumento de datos, balanceado en 8 clases (consulta de producto, servicio técnico, quejas, etc.), que captura la jerga local y el formato de mensajería instantánea del negocio.

### Conclusiones (Hito 1 — Fase Exploratoria)
* **Balance y Dificultad:** El dataset de sentimientos presenta un desbalance natural tras su agrupación, siendo la clase "Neutro" la que posee menor volumen y mayor ambigüedad semántica.
* **Correlación de Longitud:** Se comprobó que la longitud del texto es un predictor fuerte para la urgencia; los clientes insatisfechos (quejas/reclamos) escriben mensajes significativamente más largos que los clientes satisfechos o los que realizan consultas rápidas.
* **Separabilidad Léxica:** Las intenciones del negocio son semánticamente distinguibles mediante n-gramas (ej. consultas usan formato interrogativo, quejas formato declarativo). Sin embargo, existe una alta superposición léxica entre quejas y reclamos de garantía, lo que justifica el uso futuro de embeddings contextuales para su correcta clasificación.

### Conclusiones (Hito 2 — Modelización y Resultados)
* **Sentimiento (Amazon, binario):** El fine-tuning de BETO (BERT en español) obtuvo el mejor desempeño (F1-macro 0.793, ROC-AUC 0.911), superando a los modelos clásicos TF-IDF (Regresión Logística, Naive Bayes, SVM, XGBoost) aun entrenado con una muestra 11 veces menor (18k vs. 200k reseñas). Su curva de aprendizaje por época mostró que el punto óptimo fue la época 3 de 4; entrenar más allá solo aumenta el sobreajuste sin mejorar el test.
* **Intenciones (Fonazo):** El problema principal no fue el algoritmo sino el volumen de datos. Con las 96 muestras originales ningún modelo superó un F1-macro de 0.50. Al ampliar el corpus a 296 mensajes mediante variaciones controladas, el mismo SVM lineal mejoró su F1-macro en +28.4 puntos (0.499 → 0.783). El fine-tuning de BETO sobre el dataset ampliado alcanzó el F1-macro más alto (0.887), pero con sobreajuste total del conjunto de entrenamiento y evaluado sobre un test set pequeño (59 mensajes), por lo que se documenta como resultado indicativo, no concluyente.
* **Lección general:** en escenarios de datos escasos, invertir en más datos etiquetados tuvo mayor impacto y fue más confiable que cambiar de algoritmo — válido incluso al comparar contra un modelo de última generación como BETO.

### Estructura del código
```
code/
  notebooks/    EDA_Fonazo.ipynb, Modelado_Amazon.ipynb, Modelado_Fonazo.ipynb, Modelado_Amazon_BETO.ipynb
  resultados/
    figuras/    Todas las visualizaciones generadas (fig1 a fig18)
    tablas/     CSV con las métricas exactas de cada modelo y comparaciones
  app/          Interfaz gráfica (Streamlit) para stakeholders — ver sección siguiente
```

### Interfaz gráfica para stakeholders (app/)

Se incluye una app simple (Streamlit) para que un usuario de negocio (no técnico)
pueda escribir un mensaje de cliente y ver, en vivo: la intención detectada, el
sentimiento y si requiere escalamiento urgente — además de un dashboard con los
resultados de evaluación de los modelos.

**Para correrla localmente (modo clásico, sin GPU):**
```bash
cd code/app
pip install -r requirements.txt
streamlit run app.py
```
Abre automáticamente `http://localhost:8501`.

**Para habilitar el modo BETO (más preciso):** primero hay que generar los
pesos fine-tuneados ejecutando `code/notebooks/Modelado_Amazon_BETO.ipynb` y
`code/notebooks/Modelado_Fonazo.ipynb` (ambos incluyen una celda final que
guarda los pesos en `code/app/modelos/beto_amazon/` y `beto_fonazo/`). Esos
pesos no están en el repositorio (~440MB cada uno) — la app detecta
automáticamente si existen y habilita la opción "BETO" en ese caso; si no,
solo muestra el modo clásico.

Para regenerar los modelos clásicos (`amazon_logreg.joblib`,
`fonazo_svm.joblib`) desde cero: `python code/app/entrenar_modelos_clasicos.py`.

### Licencia
Este proyecto se distribuye bajo la licencia MIT.