"""Fixtures compartidas por las pruebas del MVP."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "proyecto" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from predictor import Predictor  # noqa: E402

CLASES = ("negativo", "neutral", "positivo")


class ModeloFalso:
    """Clasificador determinista para probar el contrato sin cargar MPNet."""

    classes_ = np.array(CLASES)

    def predict_proba(self, textos):
        filas = []
        for texto in textos:
            t = str(texto).lower()
            if not t.strip():
                filas.append([0.33, 0.34, 0.33])
            elif "excelente" in t or "bien" in t or "claro" in t:
                filas.append([0.05, 0.15, 0.80])
            elif "mal" in t or "no me gust" in t or "desorganiz" in t:
                filas.append([0.75, 0.20, 0.05])
            else:
                filas.append([0.20, 0.60, 0.20])
        return np.asarray(filas, dtype=float)


@pytest.fixture
def predictor_falso() -> Predictor:
    p = Predictor(verificar_hash=False)
    p._modelo = ModeloFalso()
    p._metadata = {
        "rep": "emb_mpnet",
        "clf": "rf",
        "clases": list(CLASES),
        "fecha": "2026-11-05",
        "test_accuracy": 0.69,
        "test_f1_macro": 0.68,
        "test_f1_ic95": [0.58, 0.76],
        "matriz_confusion": [[28, 9, 4], [8, 42, 2], [0, 15, 16]],
        "sklearn": "1.9.1",
        "python": "3.13.15",
    }
    p._registro = json.loads((RAIZ / "proyecto" / "modelos" / "registro.json").read_text(encoding="utf-8"))
    p.modelo_path = Path("modelo_test.joblib")
    p._version = "test-1.0"
    p._hash = "0" * 64
    p._hash_verificado = True
    p._segundos_carga = 0.0
    return p
