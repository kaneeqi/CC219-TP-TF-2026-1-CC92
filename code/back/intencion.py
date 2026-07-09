# -*- coding: utf-8 -*-
"""Clasificador de intencion (Fonazo, 8 clases). Tarea independiente de
sentimiento.py -- dataset, vectorizador y modelos propios. Permite elegir
entre los modelos clasicos o BETO v3 (fine-tuneado con datos reales,
mejor desempeno validado)."""
import os

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocesamiento import limpiar_texto

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
RUTA_BETO = os.path.join(MODELOS_DIR, "beto_fonazo_v3")

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
        _cache_clasicos["vectorizer"] = joblib.load(os.path.join(MODELOS_DIR, "fonazo_vectorizer.joblib"))
    return _cache_clasicos["vectorizer"]


def _cargar_clasico(nombre_modelo: str):
    if nombre_modelo not in _cache_clasicos:
        ruta = os.path.join(MODELOS_DIR, f"fonazo_{nombre_modelo}.joblib")
        _cache_clasicos[nombre_modelo] = joblib.load(ruta)
    return _cache_clasicos[nombre_modelo]


def _cargar_label_encoder_xgb():
    if "label_encoder_xgb" not in _cache_clasicos:
        ruta = os.path.join(MODELOS_DIR, "fonazo_label_encoder_xgb.joblib")
        _cache_clasicos["label_encoder_xgb"] = joblib.load(ruta) if os.path.exists(ruta) else None
    return _cache_clasicos["label_encoder_xgb"]


def _cargar_beto():
    if not _cache_beto:
        tokenizer = AutoTokenizer.from_pretrained(RUTA_BETO)
        modelo = AutoModelForSequenceClassification.from_pretrained(RUTA_BETO)
        modelo.to(_device).eval()
        label_encoder = joblib.load(os.path.join(RUTA_BETO, "label_encoder.joblib"))
        _cache_beto["tokenizer"] = tokenizer
        _cache_beto["modelo"] = modelo
        _cache_beto["label_encoder"] = label_encoder
    return _cache_beto["tokenizer"], _cache_beto["modelo"], _cache_beto["label_encoder"]


def _predecir_clasico(texto: str, nombre_modelo: str) -> tuple[str, float]:
    vectorizer = _cargar_vectorizer()
    clf = _cargar_clasico(nombre_modelo)
    x = vectorizer.transform([limpiar_texto(texto)])
    proba = clf.predict_proba(x)[0]
    idx = int(proba.argmax())
    etiqueta = clf.classes_[idx]
    if nombre_modelo == "xgboost":
        le_xgb = _cargar_label_encoder_xgb()
        if le_xgb is not None:
            etiqueta = le_xgb.inverse_transform([etiqueta])[0]
    return etiqueta, float(proba[idx])


def _predecir_beto(texto: str) -> tuple[str, float]:
    tokenizer, modelo, label_encoder = _cargar_beto()
    inputs = tokenizer(texto, truncation=True, padding="max_length", max_length=48, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        proba = torch.softmax(modelo(**inputs).logits, dim=1)[0]
    idx = int(torch.argmax(proba))
    etiqueta = label_encoder.inverse_transform([idx])[0]
    return etiqueta, float(proba[idx])


def predecir(texto: str, modelo: str = "beto") -> tuple[str, float]:
    """modelo: 'beto' (default, recomendado) o una clave de MODELOS_CLASICOS."""
    if modelo == "beto":
        if not os.path.isdir(RUTA_BETO):
            raise ValueError("BETO no está disponible localmente (pesos no encontrados).")
        return _predecir_beto(texto)
    if modelo not in MODELOS_CLASICOS:
        raise ValueError(f"Modelo de intención desconocido: {modelo!r}. Opciones: {list(modelos_disponibles())}")
    return _predecir_clasico(texto, modelo)
