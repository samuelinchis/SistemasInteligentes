"""Normalización de comentarios (paso 2 de la diapositiva 4).

Se deja en un módulo aparte para que el Pipeline guardado con joblib pueda
importarlo también desde el MVP.
"""
import re
import unicodedata

# Emojis/emoticonos frecuentes -> palabra con polaridad (el dataset actual no trae
# emojis, pero el MVP sí puede recibirlos).
EMOJIS = {
    "🙂": " emoji_positivo ", "😀": " emoji_positivo ", "😃": " emoji_positivo ",
    "😊": " emoji_positivo ", "👍": " emoji_positivo ", "❤": " emoji_positivo ",
    "😍": " emoji_positivo ", ":)": " emoji_positivo ", ":d": " emoji_positivo ",
    "🙁": " emoji_negativo ", "😞": " emoji_negativo ", "😠": " emoji_negativo ",
    "😡": " emoji_negativo ", "👎": " emoji_negativo ", "😢": " emoji_negativo ",
    ":(": " emoji_negativo ",
    "😐": " emoji_neutral ", "🤔": " emoji_neutral ",
}

INTENSIFICADORES = {"muy", "bastante", "demasiado", "super", "súper", "extremadamente",
                    "sumamente", "totalmente", "realmente", "algo", "poco", "tan", "más", "mas"}
NEGADORES = {"no", "nunca", "jamás", "jamas", "ni", "tampoco", "sin"}


def normalizar(texto, marcar=True):
    """Limpia un comentario.

    - minúsculas, Unicode NFKC, espacios colapsados;
    - emojis -> tokens emoji_positivo / emoji_negativo / emoji_neutral;
    - números -> <num>;
    - si ``marcar``: une intensificadores y negadores con la palabra siguiente
      ("muy claro" -> "muy_claro", "no fue" -> "no_fue") para que TF-IDF
      distinga "claro" de "no claro".
    """
    if texto is None or (isinstance(texto, float) and texto != texto):
        return ""
    t = unicodedata.normalize("NFKC", str(texto)).lower()
    for e, rep in EMOJIS.items():
        t = t.replace(e, rep)
    t = re.sub(r"\d+([.,]\d+)?", " <num> ", t)
    t = re.sub(r"[^\wáéíóúüñ<>_\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if marcar:
        tokens, salida, i = t.split(), [], 0
        while i < len(tokens):
            w = tokens[i]
            if (w in INTENSIFICADORES or w in NEGADORES) and i + 1 < len(tokens):
                salida.append(f"{w}_{tokens[i + 1]}")
                salida.append(tokens[i + 1])  # se conserva también la palabra sola
                i += 2
            else:
                salida.append(w)
                i += 1
        t = " ".join(salida)
    return t


def normalizar_basico(texto):
    return normalizar(texto, marcar=False)


def normalizar_marcado(texto):
    return normalizar(texto, marcar=True)
