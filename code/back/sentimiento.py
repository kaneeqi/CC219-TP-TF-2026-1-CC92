# -*- coding: utf-8 -*-
"""Clasificador de sentimiento (Amazon Reviews + chat). Tarea independiente
de intencion.py -- dataset, vectorizador y modelos propios. Permite elegir
entre los modelos clasicos (rapidos, sin GPU) o BETO v2 (fine-tuneado,
mejor desempeno fuera de dominio, requiere GPU/torch)."""
import os

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocesamiento import limpiar_texto

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
RUTA_BETO = os.path.join(MODELOS_DIR, "beto_amazon_v2")

MODELOS_CLASICOS = {
    "logreg": "Regresión Logística",
    "naive_bayes": "Naive Bayes Multinomial",
    "svm": "SVM Lineal",
    "xgboost": "XGBoost",
}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cache_clasicos = {}
_cache_beto = {}


def modelos_disponibles() -> dict:
    disponibles = dict(MODELOS_CLASICOS)
    if os.path.isdir(RUTA_BETO):
        disponibles["beto"] = "BETO (fine-tuned)"
    return disponibles


def _cargar_vectorizer():
    if "vectorizer" not in _cache_clasicos:
        _cache_clasicos["vectorizer"] = joblib.load(os.path.join(MODELOS_DIR, "amazon_vectorizer.joblib"))
    return _cache_clasicos["vectorizer"]


def _cargar_clasico(nombre_modelo: str):
    if nombre_modelo not in _cache_clasicos:
        ruta = os.path.join(MODELOS_DIR, f"amazon_{nombre_modelo}.joblib")
        _cache_clasicos[nombre_modelo] = joblib.load(ruta)
    return _cache_clasicos[nombre_modelo]


def _cargar_beto():
    if not _cache_beto:
        tokenizer = AutoTokenizer.from_pretrained(RUTA_BETO)
        modelo = AutoModelForSequenceClassification.from_pretrained(RUTA_BETO)
        modelo.to(_device).eval()
        _cache_beto["tokenizer"] = tokenizer
        _cache_beto["modelo"] = modelo
    return _cache_beto["tokenizer"], _cache_beto["modelo"]


def _predecir_clasico(texto: str, nombre_modelo: str) -> tuple[str, float]:
    vectorizer = _cargar_vectorizer()
    clf = _cargar_clasico(nombre_modelo)
    x = vectorizer.transform([limpiar_texto(texto)])
    proba = clf.predict_proba(x)[0]
    idx_pos = list(clf.classes_).index(1)
    es_positivo = proba[idx_pos] >= 0.5
    confianza = proba[idx_pos] if es_positivo else (1 - proba[idx_pos])
    return ("positivo" if es_positivo else "negativo"), float(confianza)


def _predecir_beto(texto: str) -> tuple[str, float]:
    tokenizer, modelo = _cargar_beto()
    inputs = tokenizer(texto, truncation=True, padding="max_length", max_length=96, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        proba = torch.softmax(modelo(**inputs).logits, dim=1)[0]
    es_positivo = proba[1] >= proba[0]
    confianza = proba[1] if es_positivo else proba[0]
    return ("positivo" if es_positivo else "negativo"), float(confianza)


def predecir(texto: str, modelo: str = "beto") -> tuple[str, float]:
    """modelo: 'beto' (default, recomendado) o una clave de MODELOS_CLASICOS."""
    if modelo == "beto":
        if not os.path.isdir(RUTA_BETO):
            raise ValueError("BETO no está disponible localmente (pesos no encontrados).")
        return _predecir_beto(texto)
    if modelo not in MODELOS_CLASICOS:
        raise ValueError(f"Modelo de sentimiento desconocido: {modelo!r}. Opciones: {list(modelos_disponibles())}")
    return _predecir_clasico(texto, modelo)
