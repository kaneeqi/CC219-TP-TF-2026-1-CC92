# -*- coding: utf-8 -*-
"""API (FastAPI) del Asistente Analítico NLP - Fonazo.

Orquesta las dos tareas independientes (sentimiento e intención, cada una en
su propio módulo con su propio selector de modelo) y aplica la lógica de
negocio que las combina: qué intenciones disparan urgencia, y para cuáles
intenciones el sentimiento no aporta valor confiable (ver hallazgo de sesgo
hacia "negativo" en mensajes puramente informativos).
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import intencion
import sentimiento

app = FastAPI(
    title="Asistente Analítico NLP — Fonazo",
    description="Clasificación de intención, sentimiento y urgencia de mensajes de clientes.",
    version="1.0",
)

# Habilitado para que un frontend separado (otro puerto/origen) pueda
# consumir la API durante desarrollo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Intenciones que, combinadas con sentimiento negativo, disparan urgencia
# (ver Mejoras_Intencion_FineTuning_BETO_v2.ipynb, sección 4).
INTENCIONES_URGENTES_POR_CONTEXTO = {
    "queja_producto_servicio", "solicitud_garantia_cambio", "seguimiento_reparacion",
}

# Intenciones puramente informativas: el modelo de sentimiento (binario,
# nunca vio ejemplos neutrales) tiende a etiquetarlas como "negativo" sin
# serlo. Para estas no se calcula ni se muestra sentimiento.
INTENCIONES_SIN_SENTIMIENTO_RELEVANTE = {
    "consulta_producto", "consulta_precio", "consulta_servicio_tecnico", "saludo_cierre",
}


class MensajeIn(BaseModel):
    texto: str
    modelo_intencion: str = "beto"
    modelo_sentimiento: str = "beto"


class ResultadoTarea(BaseModel):
    etiqueta: str
    confianza: float
    modelo: str


class AnalisisOut(BaseModel):
    texto: str
    intencion: ResultadoTarea
    sentimiento: Optional[ResultadoTarea]
    sentimiento_aplicable: bool
    urgente: bool


def _evaluar_urgencia(intencion_etiqueta: str, sentimiento_etiqueta: Optional[str]) -> bool:
    if intencion_etiqueta == "escalamiento_urgente":
        return True
    if sentimiento_etiqueta == "negativo" and intencion_etiqueta in INTENCIONES_URGENTES_POR_CONTEXTO:
        return True
    return False


@app.get("/salud")
def salud():
    return {"estado": "ok"}


@app.get("/modelos")
def modelos_disponibles():
    return {
        "sentimiento": sentimiento.modelos_disponibles(),
        "intencion": intencion.modelos_disponibles(),
    }


@app.post("/sentimiento", response_model=ResultadoTarea)
def endpoint_sentimiento(payload: MensajeIn):
    try:
        etiqueta, confianza = sentimiento.predecir(payload.texto, payload.modelo_sentimiento)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ResultadoTarea(etiqueta=etiqueta, confianza=confianza, modelo=payload.modelo_sentimiento)


@app.post("/intencion", response_model=ResultadoTarea)
def endpoint_intencion(payload: MensajeIn):
    try:
        etiqueta, confianza = intencion.predecir(payload.texto, payload.modelo_intencion)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ResultadoTarea(etiqueta=etiqueta, confianza=confianza, modelo=payload.modelo_intencion)


@app.post("/analizar", response_model=AnalisisOut)
def endpoint_analizar(payload: MensajeIn):
    try:
        intencion_etiqueta, intencion_confianza = intencion.predecir(payload.texto, payload.modelo_intencion)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sentimiento_aplicable = intencion_etiqueta not in INTENCIONES_SIN_SENTIMIENTO_RELEVANTE
    sentimiento_out = None
    sentimiento_etiqueta = None
    if sentimiento_aplicable:
        try:
            sentimiento_etiqueta, sentimiento_confianza = sentimiento.predecir(payload.texto, payload.modelo_sentimiento)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        sentimiento_out = ResultadoTarea(etiqueta=sentimiento_etiqueta, confianza=sentimiento_confianza, modelo=payload.modelo_sentimiento)

    return AnalisisOut(
        texto=payload.texto,
        intencion=ResultadoTarea(etiqueta=intencion_etiqueta, confianza=intencion_confianza, modelo=payload.modelo_intencion),
        sentimiento=sentimiento_out,
        sentimiento_aplicable=sentimiento_aplicable,
        urgente=_evaluar_urgencia(intencion_etiqueta, sentimiento_etiqueta),
    )
