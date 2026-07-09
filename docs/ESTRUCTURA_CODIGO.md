# Estructura del código

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
  docker-compose.yml   Orquesta ambos contenedores — ver puesta en marcha rápida
```

Ver también: [Puesta en marcha rápida (Docker)](../CONTRIBUTING.md)

[← Volver al README](../README.md)
