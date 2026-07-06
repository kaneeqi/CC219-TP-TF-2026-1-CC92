# -*- coding: utf-8 -*-
"""Entrena los 4 modelos clasicos evaluados en los notebooks (Regresion
Logistica, Naive Bayes, SVM Lineal, XGBoost) sobre el dataset completo y los
persiste para que la app Streamlit permita elegir cualquiera de ellos."""
import re
import time
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

STOPWORDS_ES = {
    'de', 'la', 'el', 'en', 'y', 'a', 'que', 'los', 'se', 'del', 'las', 'un', 'por',
    'una', 'con', 'no', 'es', 'su', 'al', 'lo', 'mas', 'pero', 'si', 'como', 'me',
    'le', 'mi', 'muy', 'este', 'esta', 'ya', 'tambien', 'hay', 'para', 'son', 'fue',
    'todo', 'bien', 'cuando', 'sobre', 'sin', 'han', 'he', 'o', 'e', 'ni',
    'era', 'ser', 'tiene', 'habia', 'sus', 'mismo', 'tan', 'asi',
    'unos', 'desde', 'porque', 'hasta', 'te', 'les', 'nos', 'yo', 'tu', 'tus'
}


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-zñáéíóú ]', ' ', texto)
    palabras = [w for w in texto.split() if w not in STOPWORDS_ES and len(w) > 2]
    return ' '.join(palabras)


RANDOM_STATE = 42
OUT_DIR = "D:/TF_Aplicaciones_DataScience/repo/code/app/modelos"


def entrenar_los_4(X, y, prefijo, y_binario_para_xgb=None):
    """Entrena y guarda Regresion Logistica, Naive Bayes, SVM Lineal y XGBoost."""
    modelos = {}

    t0 = time.time()
    modelos["logreg"] = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="saga", n_jobs=-1, random_state=RANDOM_STATE
    )
    modelos["logreg"].fit(X, y)
    print(f"  Regresión Logística: {time.time()-t0:.1f}s")

    t0 = time.time()
    modelos["naive_bayes"] = MultinomialNB()
    modelos["naive_bayes"].fit(X, y)
    print(f"  Naive Bayes: {time.time()-t0:.1f}s")

    t0 = time.time()
    modelos["svm"] = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE), cv=3)
    modelos["svm"].fit(X, y)
    print(f"  SVM Lineal (calibrado): {time.time()-t0:.1f}s")

    t0 = time.time()
    y_xgb = y_binario_para_xgb if y_binario_para_xgb is not None else y
    modelos["xgboost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    )
    modelos["xgboost"].fit(X, y_xgb)
    print(f"  XGBoost: {time.time()-t0:.1f}s")

    for nombre, modelo in modelos.items():
        joblib.dump(modelo, f"{OUT_DIR}/{prefijo}_{nombre}.joblib")
    print(f"  {prefijo}: 4 modelos guardados")


# ---------------------------------------------------------------------------
# Amazon: sentimiento binario (200,000 resenas)
# ---------------------------------------------------------------------------
print("--- Amazon: cargando dataset ---")
t0 = time.time()
dataset = load_dataset("SetFit/amazon_reviews_multi_es")
df = pd.DataFrame(dataset["train"])
df = df.rename(columns={"text": "review_body", "label": "stars"})
df["sentimiento"] = df["stars"].apply(lambda s: "negativo" if s <= 3 else "positivo")
df["texto_limpio"] = df["review_body"].apply(limpiar_texto)
print(f"Amazon cargado y preprocesado en {time.time()-t0:.1f}s")

vectorizer_amazon = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=5)
X_amazon = vectorizer_amazon.fit_transform(df["texto_limpio"])
y_amazon = (df["sentimiento"] == "positivo").astype(int).to_numpy()
joblib.dump(vectorizer_amazon, f"{OUT_DIR}/amazon_vectorizer.joblib")

print("Entrenando modelos de Amazon...")
entrenar_los_4(X_amazon, y_amazon, "amazon")

# ---------------------------------------------------------------------------
# Fonazo: intencion, 8 clases (dataset aumentado, 296 mensajes)
# ---------------------------------------------------------------------------
print("\n--- Fonazo: cargando dataset aumentado ---")
df_fz = pd.read_csv("D:/TF_Aplicaciones_DataScience/repo/data/intenciones_dataset_aumentado.csv")
df_fz["texto_limpio"] = df_fz["texto"].apply(limpiar_texto)

vectorizer_fonazo = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=1)
X_fonazo = vectorizer_fonazo.fit_transform(df_fz["texto_limpio"])
y_fonazo = df_fz["intencion"].astype(str).to_numpy(dtype=object)
joblib.dump(vectorizer_fonazo, f"{OUT_DIR}/fonazo_vectorizer.joblib")

# XGBoost no acepta etiquetas string directamente: se codifican a enteros y se
# guarda el encoder para poder traducir las predicciones de vuelta.
from sklearn.preprocessing import LabelEncoder
le_fonazo = LabelEncoder()
y_fonazo_int = le_fonazo.fit_transform(y_fonazo)
joblib.dump(le_fonazo, f"{OUT_DIR}/fonazo_label_encoder_xgb.joblib")

print("Entrenando modelos de Fonazo...")
entrenar_los_4(X_fonazo, y_fonazo, "fonazo", y_binario_para_xgb=y_fonazo_int)

print("\nListo. Artefactos en:", OUT_DIR)
