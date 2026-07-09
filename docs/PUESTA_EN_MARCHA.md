# Puesta en marcha rápida (Docker)

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

## Correr el backend sin Docker (desarrollo local)

```bash
cd code/back
pip install -r requirements.txt
uvicorn main:app --reload
```
Documentación interactiva (Swagger) en `http://localhost:8000/docs`. Para regenerar los modelos clásicos desde cero: `python code/back/entrenar_modelos_clasicos.py`.

## Correr el frontend sin Docker (desarrollo local)

```bash
cd code/front
npm install
npm run dev
```
Por defecto apunta a `http://localhost:8000` (backend corriendo aparte, sin Docker). Abre `http://localhost:5173`.

[← Volver al README](../README.md)
