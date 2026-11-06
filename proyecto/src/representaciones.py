"""Transformador sklearn que convierte textos en embeddings de sentence-transformers.

El modelo preentrenado no se ajusta con los datos (no hay `fit` real), así que
usarlo dentro de la validación cruzada no produce fuga de información. Se guarda
una caché en memoria para no recalcular embeddings en cada pliegue.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_MODELOS = {}
_CACHE = {}


def _cargar(nombre):
    if nombre not in _MODELOS:
        from sentence_transformers import SentenceTransformer
        _MODELOS[nombre] = SentenceTransformer(nombre, device="cpu")
    return _MODELOS[nombre]


class EmbeddingsFrase(BaseEstimator, TransformerMixin):
    def __init__(self, modelo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 normalizar=True):
        self.modelo = modelo
        self.normalizar = normalizar

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        textos = [str(x) if x is not None else "" for x in X]
        faltan = [t for t in dict.fromkeys(textos) if (self.modelo, t) not in _CACHE]
        if faltan:
            vec = _cargar(self.modelo).encode(faltan, batch_size=32, show_progress_bar=False,
                                               normalize_embeddings=self.normalizar)
            for t, v in zip(faltan, vec):
                _CACHE[(self.modelo, t)] = v
        return np.vstack([_CACHE[(self.modelo, t)] for t in textos])

    def __getstate__(self):  # al serializar con joblib no se guarda el modelo pesado
        return self.__dict__.copy()
