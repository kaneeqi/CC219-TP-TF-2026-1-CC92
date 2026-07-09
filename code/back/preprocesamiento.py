# -*- coding: utf-8 -*-
"""Limpieza de texto compartida por los modelos clasicos de ambas tareas
(sentimiento e intencion). No es un modelo, es la misma normalizacion de
texto que ya usaban ambos entrenamientos -- vive aparte para no duplicar la
lista de stopwords y el plegado de tildes en dos archivos."""
import re

STOPWORDS_ES = {
    'de', 'la', 'el', 'en', 'y', 'a', 'que', 'los', 'se', 'del', 'las', 'un', 'por',
    'una', 'con', 'no', 'es', 'su', 'al', 'lo', 'mas', 'pero', 'si', 'como', 'me',
    'le', 'mi', 'muy', 'este', 'esta', 'ya', 'tambien', 'hay', 'para', 'son', 'fue',
    'todo', 'bien', 'cuando', 'sobre', 'sin', 'han', 'he', 'o', 'e', 'ni',
    'era', 'ser', 'tiene', 'habia', 'sus', 'mismo', 'tan', 'asi',
    'unos', 'desde', 'porque', 'hasta', 'te', 'les', 'nos', 'yo', 'tu', 'tus'
}
ACENTOS = str.maketrans("áéíóú", "aeiou")


def limpiar_texto(texto: str) -> str:
    texto = str(texto).lower().translate(ACENTOS)
    texto = re.sub(r"[^a-zñ ]", " ", texto)
    palabras = [w for w in texto.split() if w not in STOPWORDS_ES and len(w) > 2]
    return " ".join(palabras)
