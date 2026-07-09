FROM python:3.12-slim

WORKDIR /app

# Torch CPU-only: la inferencia (un mensaje a la vez) no necesita GPU, y la
# imagen queda mucho mas liviana que con la variante CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py sentimiento.py intencion.py preprocesamiento.py ./

# Los pesos de los modelos (modelos/) NO se copian a la imagen -- se montan
# como volumen en runtime (ver README de esta carpeta), para no duplicar
# ~1.3GB de checkpoints de BETO en cada build.

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
