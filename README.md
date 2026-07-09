# Asistente Analítico NLP para E-commerce (Fonazo)

### Objetivo del trabajo
Desarrollar un Asistente Analítico basado en Procesamiento de Lenguaje Natural (NLP) capaz de automatizar la categorización de mensajes no estructurados de clientes. El sistema clasificará la intención del usuario (multiclase), el sentimiento del texto (polaridad) y predecirá la necesidad de escalamiento urgente a un asesor humano.

### Nombres de los alumnos participantes
* Valdivia Guzman, Jose Agustin (U201822153)
* Bravo Lévano, Eduardo Fernando (U20231E122)
* Cunyas Ramos, Iván Rubén (U202316818)

### Breve descripción del dataset
El proyecto utiliza un enfoque de dataset híbrido compuesto por dos fuentes:
1. **Dataset de Sentimientos (Amazon Reviews ES):** ~200,000 reseñas de e-commerce mapeadas a 2 clases de sentimiento (Positivo, Negativo), complementado con un corpus sintético de 400 mensajes de chat (formato de atención al cliente, distinto al registro de reseña) y 5 reseñas reales de tiendas peruanas de accesorios/reparación Apple, para validar y corregir el desempeño fuera del dominio de entrenamiento original. *(Ver informe adjunto para más detalles).*
2. **Dataset de Intenciones (Fonazo):** Corpus de 96 registros originales, ampliado a 296 mediante aumento de datos (variaciones de vocabulario), y posteriormente a 564 con variantes de puntuación y tildes (para no depender de si el cliente usa `¿` de apertura o acentúa correctamente, poco realista en WhatsApp). Se sumaron además **143 mensajes reales** escritos por el equipo (sin plantilla, para capturar variedad genuina de redacción), de los cuales 115 se usaron para entrenamiento (dataset final: 679 registros, `intenciones_dataset_v3.csv`) y 28 se reservaron como conjunto de validación nunca visto por ningún modelo (`intenciones_holdout_real.csv`). Balanceado en 8 clases (consulta de producto, precio, servicio técnico, quejas, garantía, seguimiento de reparación, escalamiento urgente y saludo/cierre), capturando la jerga local y el formato de mensajería instantánea del negocio.

### Conclusiones (Hito 1 — Fase Exploratoria)
* **Balance y Dificultad:** El dataset de sentimientos presenta un desbalance natural tras su agrupación, siendo la clase "Neutro" la que posee menor volumen y mayor ambigüedad semántica.
* **Correlación de Longitud:** Se comprobó que la longitud del texto es un predictor fuerte para la urgencia; los clientes insatisfechos (quejas/reclamos) escriben mensajes significativamente más largos que los clientes satisfechos o los que realizan consultas rápidas.
* **Separabilidad Léxica:** Las intenciones del negocio son semánticamente distinguibles mediante n-gramas (ej. consultas usan formato interrogativo, quejas formato declarativo). Sin embargo, existe una alta superposición léxica entre quejas y reclamos de garantía, lo que justifica el uso futuro de embeddings contextuales para su correcta clasificación.

### Conclusiones (Hito 2 — Modelización y Resultados)
* **Sentimiento (Amazon, binario):** El fine-tuning de BETO (BERT en español) obtuvo el mejor desempeño (F1-macro 0.793, ROC-AUC 0.911), superando a los modelos clásicos TF-IDF (Regresión Logística, Naive Bayes, SVM, XGBoost) aun entrenado con una muestra 11 veces menor (18k vs. 200k reseñas). Su curva de aprendizaje por época mostró que el punto óptimo fue la época 3 de 4; entrenar más allá solo aumenta el sobreajuste sin mejorar el test. Al validar este modelo contra 70 mensajes de chat reales en formato (no reseñas), su desempeño cayó a F1-macro 0.823, fallando sistemáticamente en positivos cortos y transaccionales ("gracias por la garantía, me cambiaron el producto sin problema") — había aprendido el registro de reseña, no el de atención al cliente. Se corrigió con un fine-tuning combinado (Amazon + 400 mensajes de chat), alcanzando F1-macro 1.0 en el mismo set de validación, y validado además contra reseñas reales de tiendas peruanas del rubro (5/5 correctas).
* **Intenciones (Fonazo):** El problema principal no fue el algoritmo sino el volumen de datos. Con las 96 muestras originales ningún modelo superó un F1-macro de 0.50. Al ampliar el corpus a 296 mensajes mediante variaciones controladas, el mismo SVM lineal mejoró su F1-macro en +28.4 puntos (0.499 → 0.783). El fine-tuning de BETO sobre el dataset ampliado alcanzó el F1-macro más alto (0.887), pero con sobreajuste total del conjunto de entrenamiento y evaluado sobre un test set pequeño (59 mensajes), por lo que se documentó como resultado indicativo, no concluyente — confirmado luego al validar contra mensajes fuera de plantilla, donde cayó a 0.716. Esa misma prueba reveló que el corpus dependía en parte de una señal superficial (casi todas las preguntas de entrenamiento usaban el signo de apertura `¿`, poco común en WhatsApp real). Corregido el preprocesamiento (normalización de tildes) y ampliado el corpus con variantes de puntuación, el F1-macro fuera de plantilla subió a 0.873 (BETO). Se evaluó fusionar clases con alta confusión aparente (`queja_producto_servicio` / `solicitud_garantia_cambio`) y eliminar `escalamiento_urgente`; ambas hipótesis se descartaron con evidencia empírica — el sesgo de puntuación explicaba casi toda la confusión entre las dos primeras, y `escalamiento_urgente` resultó ser una de las clases mejor clasificadas y no redundante con las demás. Sí se amplió la regla de negocio de urgencia para incluir `seguimiento_reparacion` con sentimiento negativo. Finalmente, se integraron 143 mensajes reales escritos por el equipo (crowdsourcing), subiendo el F1-macro de BETO a 0.901 (set sintético) y **0.888 sobre un holdout real de 28 mensajes nunca usados en entrenamiento** — la validación más confiable del proyecto, con mejora consistente en los 5 modelos evaluados (clásicos y BETO).
* **Lección general:** en escenarios de datos escasos, invertir en más datos etiquetados tuvo mayor impacto y fue más confiable que cambiar de algoritmo — válido incluso al comparar contra un modelo de última generación como BETO. Además, validar contra datos fuera del dominio/formato de entrenamiento fue indispensable: los números de benchmark inicial (incluso los "buenos") escondían sobreajuste y sesgos de formato que solo aparecieron al probar con texto en el registro real de uso.

El detalle completo de esta segunda ronda de validación y corrección, con
matrices de confusión y curvas de entrenamiento train/eval por época, está en
los notebooks `Mejoras_Sentimiento_v2.ipynb`, `Mejoras_Intencion_Preprocesamiento.ipynb`,
`Mejoras_Intencion_FineTuning_BETO_v2.ipynb` y `Mejoras_Intencion_DatosReales_v3.ipynb`.

### Estructura del código
```
code/
  notebooks/
    EDA_Fonazo.ipynb, Modelado_Amazon.ipynb, Modelado_Fonazo.ipynb,
    Modelado_Amazon_BETO.ipynb          Hito 2 (baseline original)
    Mejoras_Sentimiento_v2.ipynb                    Validación fuera de dominio + fine-tuning BETO v2
    Mejoras_Intencion_Preprocesamiento.ipynb        Sesgo de puntuación/tildes + fix
    Mejoras_Intencion_FineTuning_BETO_v2.ipynb      Fine-tuning v2 + análisis de fusión de clases
    Mejoras_Intencion_DatosReales_v3.ipynb          Integración de 143 mensajes reales + validación final
  resultados/
    figuras/    Todas las visualizaciones generadas (fig1 a fig27)
    tablas/     CSV con las métricas exactas de cada modelo y comparaciones
  app/          Modelos entrenados (modelos/) y script de reentrenamiento clásico
```

### Modelos entrenados (app/modelos/)

Los pesos y artefactos de los modelos entrenados (vectorizadores TF-IDF,
clasificadores clásicos `.joblib`, checkpoints de BETO fine-tuneado) viven en
`code/app/modelos/`. La interfaz de usuario (API + Docker) está en desarrollo —
por ahora estos modelos se consumen directamente desde los notebooks.

Para regenerar los modelos clásicos (`amazon_logreg.joblib`,
`fonazo_svm.joblib`) desde cero: `python code/app/entrenar_modelos_clasicos.py`.

### Licencia
Este proyecto se distribuye bajo la licencia MIT.