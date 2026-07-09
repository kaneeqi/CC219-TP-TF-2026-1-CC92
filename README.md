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

<details>
<summary><b>Detalle de los archivos de datos</b> (data/)</summary>

| Archivo | Contenido |
|---|---|
| `intenciones_dataset.csv` | Corpus original (96 mensajes) |
| `intenciones_dataset_aumentado.csv` | Ampliado por variaciones de vocabulario (296) |
| `intenciones_dataset_v2.csv` | + variantes de puntuación/tildes (564) |
| `intenciones_dataset_v3.csv` | + 115 mensajes reales de entrenamiento (679, dataset final) |
| `intenciones_holdout_real.csv` | 28 mensajes reales, nunca usados en entrenamiento |
| `sentimiento_resenas_reales_dominio.csv` | 5 reseñas reales de tiendas del rubro |

</details>

### Conclusiones (Hito 1 — Fase Exploratoria)
* **Balance y Dificultad:** El dataset de sentimientos presenta un desbalance natural tras su agrupación, siendo la clase "Neutro" la que posee menor volumen y mayor ambigüedad semántica.
* **Correlación de Longitud:** Se comprobó que la longitud del texto es un predictor fuerte para la urgencia; los clientes insatisfechos (quejas/reclamos) escriben mensajes significativamente más largos que los clientes satisfechos o los que realizan consultas rápidas.
* **Separabilidad Léxica:** Las intenciones del negocio son semánticamente distinguibles mediante n-gramas (ej. consultas usan formato interrogativo, quejas formato declarativo). Sin embargo, existe una alta superposición léxica entre quejas y reclamos de garantía, lo que justifica el uso futuro de embeddings contextuales para su correcta clasificación.

### Conclusiones (Hito 2 — Modelización y Resultados)
* **Sentimiento (Amazon, binario):** El fine-tuning de BETO obtuvo el mejor desempeño inicial (F1-macro 0.793, ROC-AUC 0.911), superando a los modelos clásicos TF-IDF aun entrenado con una muestra 11 veces menor. Validado luego contra mensajes de chat reales (no reseñas), su desempeño cayó a F1-macro 0.823 — había aprendido el registro de reseña, no el de atención al cliente. Corregido con fine-tuning combinado (Amazon + chat), alcanzó F1-macro 1.0 en esa validación.
* **Intenciones (Fonazo):** El problema principal no fue el algoritmo sino el volumen de datos: con 96 muestras ningún modelo superó F1-macro 0.50; al ampliar a 296, el mismo SVM mejoró +28.4 puntos (0.499 → 0.783). El corpus además dependía de una señal superficial (signo `¿` de apertura, poco común en WhatsApp real); corregido el preprocesamiento y ampliado con datos reales de crowdsourcing, BETO alcanzó **F1-macro 0.888 sobre un holdout real nunca visto en entrenamiento** — la validación más confiable del proyecto.
* **Lección general:** en escenarios de datos escasos, invertir en más datos etiquetados (reales, no solo sintéticos) tuvo mayor impacto que cambiar de algoritmo. Y validar contra datos fuera del dominio/formato de entrenamiento fue indispensable: los números de benchmark inicial escondían sobreajuste y sesgos de formato que solo aparecieron al probar con texto en el registro real de uso.

<details>
<summary><b>Ver detalle completo de la validación y las correcciones aplicadas</b></summary>

**Sentimiento:** al validar BETO (benchmark inicial) contra 70 mensajes de chat reales en formato (no reseñas), falló sistemáticamente en positivos cortos y transaccionales ("gracias por la garantía, me cambiaron el producto sin problema"). Se corrigió con un fine-tuning combinado (Amazon + 400 mensajes de chat sintético), alcanzando F1-macro 1.0 en el mismo set, y validado además contra reseñas reales de tiendas peruanas del rubro (5/5 correctas).

**Intenciones:** el fine-tuning de BETO sobre el dataset ampliado (296) alcanzó F1-macro 0.887, pero con sobreajuste total del conjunto de entrenamiento — confirmado al validar contra mensajes fuera de plantilla, donde cayó a 0.716. Esa prueba reveló que el corpus dependía en parte de una señal superficial (casi todas las preguntas de entrenamiento usaban `¿` de apertura). Corregido el preprocesamiento (normalización de tildes) y ampliado el corpus con variantes de puntuación, el F1-macro fuera de plantilla subió a 0.873 (BETO). Se evaluó fusionar clases con alta confusión aparente (`queja_producto_servicio` / `solicitud_garantia_cambio`) y eliminar `escalamiento_urgente`; ambas hipótesis se descartaron con evidencia empírica — el sesgo de puntuación explicaba casi toda la confusión entre las dos primeras, y `escalamiento_urgente` resultó ser una de las clases mejor clasificadas y no redundante con las demás. Sí se amplió la regla de negocio de urgencia para incluir `seguimiento_reparacion` con sentimiento negativo. Finalmente, se integraron 143 mensajes reales escritos por el equipo (crowdsourcing), subiendo el F1-macro de BETO a 0.901 (set sintético) y 0.888 sobre el holdout real, con mejora consistente en los 5 modelos evaluados (clásicos y BETO).

El detalle completo, con matrices de confusión y curvas de entrenamiento train/eval por época, está en los notebooks `Mejoras_Sentimiento_v2.ipynb`, `Mejoras_Intencion_Preprocesamiento.ipynb`, `Mejoras_Intencion_FineTuning_BETO_v2.ipynb` y `Mejoras_Intencion_DatosReales_v3.ipynb`.

</details>

<details>
<summary><b>Estructura del código</b></summary>

```
code/
  notebooks/
    EDA_Fonazo.ipynb, Modelado_Amazon.ipynb, Modelado_Fonazo.ipynb,
    Modelado_Amazon_BETO.ipynb                      Hito 2 (baseline original)
    Mejoras_Sentimiento_v2.ipynb                    Validación fuera de dominio + fine-tuning BETO v2
    Mejoras_Intencion_Preprocesamiento.ipynb        Sesgo de puntuación/tildes + fix
    Mejoras_Intencion_FineTuning_BETO_v2.ipynb      Fine-tuning v2 + análisis de fusión de clases
    Mejoras_Intencion_DatosReales_v3.ipynb          Integración de 143 mensajes reales + validación final
  resultados/
    figuras/    Todas las visualizaciones generadas (fig1 a fig27)
    tablas/     CSV con las métricas exactas de cada modelo y comparaciones
  back/         API (FastAPI) + modelos entrenados
  front/        Interfaz de chat (React + Vite)
  docker-compose.yml   Orquesta ambos contenedores — ver "Puesta en marcha rápida"
```

</details>

### Puesta en marcha rápida (Docker)

**Requisitos:** [Docker](https://docs.docker.com/get-docker/) y Docker Compose (incluido en Docker Desktop). No hace falta instalar Python ni Node — todo corre dentro de los contenedores.

**1. Levantar los dos contenedores (back + front):**
```bash
cd code
docker compose up --build
```
- Frontend (chat): [http://localhost:5173](http://localhost:5173)
- Backend / Swagger interactivo: [http://localhost:8000/docs](http://localhost:8000/docs)

Con esto ya funciona todo usando los **modelos clásicos** (Regresión Logística, Naive Bayes, SVM, XGBoost) — no requieren pesos adicionales, están incluidos en el repo (`code/back/modelos/*.joblib`).

**2. (Opcional) Habilitar BETO** — el modelo con mejor desempeño validado, pero sus pesos (~420MB c/u) no se incluyen en el repositorio por tamaño. Se generan localmente:

| Modelo | Notebook que lo genera | Carpeta de salida |
|---|---|---|
| Sentimiento | `code/notebooks/Mejoras_Sentimiento_v2.ipynb` | `code/back/modelos/beto_amazon_v2/` |
| Intención | `code/notebooks/Mejoras_Intencion_DatosReales_v3.ipynb` | `code/back/modelos/beto_fonazo_v3/` |

```bash
cd code/back
pip install -r requirements.txt          # o: pip install jupyter pandas torch transformers scikit-learn xgboost datasets
jupyter nbconvert --to notebook --execute --inplace ../notebooks/Mejoras_Sentimiento_v2.ipynb
jupyter nbconvert --to notebook --execute --inplace ../notebooks/Mejoras_Intencion_DatosReales_v3.ipynb
```
Funciona con o sin GPU (más lento en CPU). Una vez generadas esas dos carpetas dentro de `code/back/modelos/`, basta con volver a levantar el compose (`docker compose up --build`, paso 1) — la API detecta los pesos automáticamente y habilita "BETO" como opción en el selector de modelos del chat.

**Apagar los contenedores:** `docker compose down` (desde `code/`).

<details>
<summary><b>Correr el backend sin Docker (desarrollo local)</b></summary>

```bash
cd code/back
pip install -r requirements.txt
uvicorn main:app --reload
```
Documentación interactiva (Swagger) en `http://localhost:8000/docs`. Para regenerar los modelos clásicos desde cero: `python code/back/entrenar_modelos_clasicos.py`.

</details>

<details>
<summary><b>Correr el frontend sin Docker (desarrollo local)</b></summary>

```bash
cd code/front
npm install
npm run dev
```
Por defecto apunta a `http://localhost:8000` (backend corriendo aparte, sin Docker). Abre `http://localhost:5173`.

</details>

### Licencia
Este proyecto se distribuye bajo la licencia MIT.
