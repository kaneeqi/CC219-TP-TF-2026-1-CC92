# Asistente Analítico NLP para E-commerce (Fonazo)

### Objetivo del trabajo
Desarrollar un Asistente Analítico basado en Procesamiento de Lenguaje Natural (NLP) capaz de automatizar la categorización de mensajes no estructurados de clientes. El sistema clasificará la intención del usuario (multiclase), el sentimiento del texto (polaridad) y predecirá la necesidad de escalamiento urgente a un asesor humano.

### Nombres de los alumnos participantes
* Valdivia Guzman, Jose Agustin (U201822153)
* Bravo Lévano, Eduardo Fernando (U20231E122)
* Cunyas Ramos, Iván Rubén (U202316818)

### Breve descripción del dataset
El proyecto utiliza un enfoque de dataset híbrido compuesto por dos fuentes:
1. **Dataset de Sentimientos (Amazon Reviews ES):** ~200,000 reseñas de e-commerce mapeadas a 2 clases de sentimiento (Positivo, Negativo), complementado con un corpus sintético de 400 mensajes de chat y 5 reseñas reales de tiendas peruanas de accesorios/reparación Apple, para validar y corregir el desempeño fuera del dominio de entrenamiento original.
2. **Dataset de Intenciones (Fonazo):** Corpus de 96 registros originales, ampliado a 296 mediante aumento de datos, y posteriormente a 564 con variantes de puntuación y tildes (poco realista depender de `¿` de apertura o acentuación correcta en WhatsApp). Se sumaron además **143 mensajes reales** escritos por el equipo (115 para entrenamiento — dataset final de 679 registros — y 28 como holdout de validación nunca visto por ningún modelo). Balanceado en 8 clases (consulta de producto, precio, servicio técnico, quejas, garantía, seguimiento de reparación, escalamiento urgente y saludo/cierre).

Detalle de cada archivo: [docs/DATASET_DETALLE.md](docs/DATASET_DETALLE.md)

### Conclusiones (Hito 1 — Fase Exploratoria)
* **Balance y Dificultad:** El dataset de sentimientos presenta un desbalance natural tras su agrupación, siendo la clase "Neutro" la que posee menor volumen y mayor ambigüedad semántica.
* **Correlación de Longitud:** Se comprobó que la longitud del texto es un predictor fuerte para la urgencia; los clientes insatisfechos (quejas/reclamos) escriben mensajes significativamente más largos que los clientes satisfechos o los que realizan consultas rápidas.
* **Separabilidad Léxica:** Las intenciones del negocio son semánticamente distinguibles mediante n-gramas (ej. consultas usan formato interrogativo, quejas formato declarativo). Sin embargo, existe una alta superposición léxica entre quejas y reclamos de garantía, lo que justifica el uso futuro de embeddings contextuales para su correcta clasificación.

### Conclusiones (Hito 2 — Modelización y Resultados)
* **Sentimiento (Amazon, binario):** El fine-tuning de BETO obtuvo el mejor desempeño inicial (F1-macro 0.793, ROC-AUC 0.911), superando a los modelos clásicos TF-IDF aun entrenado con una muestra 11 veces menor. Validado luego contra mensajes de chat reales (no reseñas), su desempeño cayó a F1-macro 0.823 — había aprendido el registro de reseña, no el de atención al cliente. Corregido con fine-tuning combinado (Amazon + chat), alcanzó F1-macro 1.0 en esa validación.
* **Intenciones (Fonazo):** El problema principal no fue el algoritmo sino el volumen de datos: con 96 muestras ningún modelo superó F1-macro 0.50; al ampliar a 296, el mismo SVM mejoró +28.4 puntos (0.499 → 0.783). El corpus además dependía de una señal superficial (signo `¿` de apertura, poco común en WhatsApp real); corregido el preprocesamiento y ampliado con datos reales de crowdsourcing, BETO alcanzó **F1-macro 0.888 sobre un holdout real nunca visto en entrenamiento** — la validación más confiable del proyecto.
* **Lección general:** en escenarios de datos escasos, invertir en más datos etiquetados (reales, no solo sintéticos) tuvo mayor impacto que cambiar de algoritmo. Y validar contra datos fuera del dominio/formato de entrenamiento fue indispensable: los números de benchmark inicial escondían sobreajuste y sesgos de formato que solo aparecieron al probar con texto en el registro real de uso.

Detalle completo de la validación y las correcciones aplicadas: [docs/MEJORAS_DETALLE.md](docs/MEJORAS_DETALLE.md)

### Licencia
Este proyecto se distribuye bajo la licencia MIT.

---

**Documentación técnica adicional:**
* [Estructura del código](docs/ESTRUCTURA_CODIGO.md)
* [Puesta en marcha rápida (Docker)](docs/PUESTA_EN_MARCHA.md)
