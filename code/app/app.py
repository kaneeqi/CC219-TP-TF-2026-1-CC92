# -*- coding: utf-8 -*-
"""Asistente Analitico NLP - Fonazo.

App Streamlit para stakeholders: permite ingresar un mensaje de cliente y ver
la intencion, el sentimiento y una bandera de urgencia predichos, eligiendo
libremente qué modelo usar para cada una de las dos tareas (son
clasificadores independientes, entrenados sobre datasets distintos), ademas
de un dashboard con los resultados de evaluacion de los modelos entrenados.
"""
import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
RESULTADOS_DIR = os.path.join(BASE_DIR, "..", "resultados")

STOPWORDS_ES = {
    'de', 'la', 'el', 'en', 'y', 'a', 'que', 'los', 'se', 'del', 'las', 'un', 'por',
    'una', 'con', 'no', 'es', 'su', 'al', 'lo', 'mas', 'pero', 'si', 'como', 'me',
    'le', 'mi', 'muy', 'este', 'esta', 'ya', 'tambien', 'hay', 'para', 'son', 'fue',
    'todo', 'bien', 'cuando', 'sobre', 'sin', 'han', 'he', 'o', 'e', 'ni',
    'era', 'ser', 'tiene', 'habia', 'sus', 'mismo', 'tan', 'asi',
    'unos', 'desde', 'porque', 'hasta', 'te', 'les', 'nos', 'yo', 'tu', 'tus'
}

INTENCIONES_URGENTES_POR_CONTEXTO = {"queja_producto_servicio", "solicitud_garantia_cambio"}

MODELOS_CLASICOS = {
    "Regresión Logística": "logreg",
    "Naive Bayes Multinomial": "naive_bayes",
    "SVM Lineal": "svm",
    "XGBoost": "xgboost",
}


def limpiar_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"[^a-zñáéíóú ]", " ", texto)
    palabras = [w for w in texto.split() if w not in STOPWORDS_ES and len(w) > 2]
    return " ".join(palabras)


@st.cache_resource
def cargar_vectorizadores():
    vec_amazon = joblib.load(os.path.join(MODELOS_DIR, "amazon_vectorizer.joblib"))
    vec_fonazo = joblib.load(os.path.join(MODELOS_DIR, "fonazo_vectorizer.joblib"))
    return vec_amazon, vec_fonazo


@st.cache_resource
def cargar_modelo_clasico(prefijo: str, sufijo: str):
    return joblib.load(os.path.join(MODELOS_DIR, f"{prefijo}_{sufijo}.joblib"))


@st.cache_resource
def cargar_label_encoder_fonazo_xgb():
    ruta = os.path.join(MODELOS_DIR, "fonazo_label_encoder_xgb.joblib")
    return joblib.load(ruta) if os.path.exists(ruta) else None


def beto_disponible() -> tuple[bool, bool]:
    amazon_ok = os.path.isdir(os.path.join(MODELOS_DIR, "beto_amazon"))
    fonazo_ok = os.path.isdir(os.path.join(MODELOS_DIR, "beto_fonazo"))
    return amazon_ok, fonazo_ok


@st.cache_resource
def cargar_beto_amazon():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ruta = os.path.join(MODELOS_DIR, "beto_amazon")
    tokenizer = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
    modelo.eval()
    return tokenizer, modelo


@st.cache_resource
def cargar_beto_fonazo():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ruta = os.path.join(MODELOS_DIR, "beto_fonazo")
    tokenizer = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
    modelo.eval()
    label_encoder = joblib.load(os.path.join(ruta, "label_encoder.joblib"))
    return tokenizer, modelo, label_encoder


def predecir_sentimiento_clasico(texto, vectorizer, clf, es_xgboost=False):
    x = vectorizer.transform([limpiar_texto(texto)])
    proba = clf.predict_proba(x)[0]
    clases = list(clf.classes_)
    idx_pos = clases.index(1)
    es_positivo = proba[idx_pos] >= 0.5
    confianza = proba[idx_pos] if es_positivo else (1 - proba[idx_pos])
    return ("positivo" if es_positivo else "negativo"), float(confianza)


def predecir_intencion_clasico(texto, vectorizer, clf, label_encoder_xgb=None):
    x = vectorizer.transform([limpiar_texto(texto)])
    proba = clf.predict_proba(x)[0]
    idx = int(proba.argmax())
    pred = clf.classes_[idx]
    if label_encoder_xgb is not None and isinstance(pred, (int, np.integer)):
        pred = label_encoder_xgb.inverse_transform([pred])[0]
    return pred, float(proba[idx])


def predecir_sentimiento_beto(texto, tokenizer, modelo):
    import torch
    inputs = tokenizer(texto, truncation=True, padding="max_length", max_length=96, return_tensors="pt")
    with torch.no_grad():
        logits = modelo(**inputs).logits
        proba = torch.softmax(logits, dim=1)[0]
    es_positivo = proba[1] >= proba[0]
    confianza = proba[1] if es_positivo else proba[0]
    return ("positivo" if es_positivo else "negativo"), float(confianza)


def predecir_intencion_beto(texto, tokenizer, modelo, label_encoder):
    import torch
    inputs = tokenizer(texto, truncation=True, padding="max_length", max_length=48, return_tensors="pt")
    with torch.no_grad():
        logits = modelo(**inputs).logits
        proba = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(proba))
    pred = label_encoder.inverse_transform([idx])[0]
    return pred, float(proba[idx])


def evaluar_urgencia(intencion: str, sentimiento: str) -> bool:
    if intencion == "escalamiento_urgente":
        return True
    if sentimiento == "negativo" and intencion in INTENCIONES_URGENTES_POR_CONTEXTO:
        return True
    return False


st.set_page_config(page_title="Asistente Analitico Fonazo", page_icon="📊", layout="wide")

st.title("📊 Asistente Analítico NLP — Fonazo")
st.caption(
    "Clasificación automática de mensajes de clientes: intención, sentimiento y "
    "necesidad de escalamiento urgente."
)

tab_demo, tab_dashboard = st.tabs(["🔍 Analizar mensaje", "📈 Resultados de los modelos"])

with tab_demo:
    beto_amazon_ok, beto_fonazo_ok = beto_disponible()

    opciones_sentimiento = list(MODELOS_CLASICOS.keys()) + (["BETO (fine-tuned)"] if beto_amazon_ok else [])
    opciones_intencion = list(MODELOS_CLASICOS.keys()) + (["BETO (fine-tuned)"] if beto_fonazo_ok else [])

    st.sidebar.markdown("### Modelos a usar")
    st.sidebar.caption("Son dos tareas independientes (datasets distintos): elige el modelo para cada una.")
    modelo_sentimiento = st.sidebar.selectbox("Modelo de sentimiento (Amazon)", opciones_sentimiento, index=0)
    modelo_intencion = st.sidebar.selectbox("Modelo de intención (Fonazo)", opciones_intencion, index=0)

    if not beto_amazon_ok or not beto_fonazo_ok:
        st.sidebar.info(
            "BETO no aparece como opción si sus pesos no están disponibles localmente "
            "(no se incluyen en el despliegue público por su tamaño)."
        )

    texto = st.text_area(
        "Mensaje del cliente",
        placeholder="Ej: Hola, compré un iPhone 14 hace una semana y la pantalla ya se rayó, quiero que me lo cambien",
        height=120,
    )

    if st.button("Analizar", type="primary") and texto.strip():
        vec_amazon, vec_fonazo = cargar_vectorizadores()

        if modelo_sentimiento == "BETO (fine-tuned)":
            tok_a, mod_a = cargar_beto_amazon()
            sentimiento, conf_sent = predecir_sentimiento_beto(texto, tok_a, mod_a)
        else:
            clf_amazon = cargar_modelo_clasico("amazon", MODELOS_CLASICOS[modelo_sentimiento])
            sentimiento, conf_sent = predecir_sentimiento_clasico(texto, vec_amazon, clf_amazon)

        if modelo_intencion == "BETO (fine-tuned)":
            tok_f, mod_f, le_f = cargar_beto_fonazo()
            intencion, conf_int = predecir_intencion_beto(texto, tok_f, mod_f, le_f)
        else:
            clf_fonazo = cargar_modelo_clasico("fonazo", MODELOS_CLASICOS[modelo_intencion])
            le_xgb = cargar_label_encoder_fonazo_xgb() if modelo_intencion == "XGBoost" else None
            intencion, conf_int = predecir_intencion_clasico(texto, vec_fonazo, clf_fonazo, le_xgb)

        urgente = evaluar_urgencia(intencion, sentimiento)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Intención detectada** · _{modelo_intencion}_")
            st.markdown(f"#### {intencion.replace('_', ' ')}")
            if conf_int is not None:
                st.progress(min(max(conf_int, 0.0), 1.0), text=f"Confianza: {conf_int:.0%}")
        with col2:
            emoji_sent = "🟢" if sentimiento == "positivo" else "🔴"
            st.markdown(f"**Sentimiento** · _{modelo_sentimiento}_")
            st.markdown(f"#### {emoji_sent} {sentimiento}")
            st.progress(min(max(conf_sent, 0.0), 1.0), text=f"Confianza: {conf_sent:.0%}")
        with col3:
            if urgente:
                st.error("🚨 Requiere atención humana inmediata")
            else:
                st.success("✅ Puede atenderse en el flujo normal")

with tab_dashboard:
    st.subheader("Comparación de modelos — Sentimiento (Amazon Reviews ES)")
    try:
        df_amazon = pd.read_csv(os.path.join(RESULTADOS_DIR, "tablas", "resultados_amazon_final_con_beto.csv"))
        st.dataframe(df_amazon, use_container_width=True)
    except FileNotFoundError:
        st.warning("Tabla de resultados de Amazon no encontrada.")

    col_a, col_b = st.columns(2)
    with col_a:
        ruta_fig = os.path.join(RESULTADOS_DIR, "figuras", "fig15_comparacion_final_amazon_con_beto.png")
        if os.path.exists(ruta_fig):
            st.image(ruta_fig, caption="Comparación final de modelos — Amazon")
    with col_b:
        ruta_fig = os.path.join(RESULTADOS_DIR, "figuras", "fig16b_curva_aprendizaje_beto_amazon.png")
        if os.path.exists(ruta_fig):
            st.image(ruta_fig, caption="Curva de aprendizaje — BETO Amazon")

    st.divider()
    st.subheader("Comparación de modelos — Intenciones (Fonazo)")
    try:
        df_fonazo = pd.read_csv(os.path.join(RESULTADOS_DIR, "tablas", "comparacion_final_fonazo_con_beto.csv"))
        st.dataframe(df_fonazo, use_container_width=True)
    except FileNotFoundError:
        st.warning("Tabla de resultados de Fonazo no encontrada.")

    col_c, col_d = st.columns(2)
    with col_c:
        ruta_fig = os.path.join(RESULTADOS_DIR, "figuras", "fig13_comparacion_aumento_datos_fonazo.png")
        if os.path.exists(ruta_fig):
            st.image(ruta_fig, caption="Efecto del aumento de datos — Fonazo")
    with col_d:
        ruta_fig = os.path.join(RESULTADOS_DIR, "figuras", "fig18_curva_aprendizaje_beto_fonazo.png")
        if os.path.exists(ruta_fig):
            st.image(ruta_fig, caption="Curva de aprendizaje — BETO Fonazo")

    st.divider()
    st.subheader("Hallazgos principales")
    st.markdown(
        """
- **Sentimiento (Amazon):** BETO obtuvo el mejor desempeño (F1-macro 0.79, ROC-AUC 0.91),
  aun entrenado con una muestra 11 veces menor que los modelos clásicos.
- **Intenciones (Fonazo):** el aumento de datos (96 → 296 mensajes) mejoró el F1-macro del
  SVM lineal en +28 puntos — el volumen de datos importó más que el algoritmo.
- BETO también ganó en Fonazo, pero mostró sobreajuste total del conjunto de entrenamiento;
  ese resultado se documenta como indicativo, no concluyente.
        """
    )
