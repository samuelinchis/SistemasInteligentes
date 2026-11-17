"""Prueba de integración con el joblib y MPNet reales.

Se ejecuta solo con RUN_INTEGRATION=1 porque la primera vez descarga el modelo
de embeddings (~1 GB):

    RUN_INTEGRATION=1 python -m pytest proyecto/tests/test_integracion.py -q
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integracion

if os.getenv("RUN_INTEGRATION") != "1":
    pytest.skip("Defina RUN_INTEGRATION=1 para probar el modelo real.", allow_module_level=True)

pytest.importorskip("sentence_transformers")

from predictor import Predictor  # noqa: E402


@pytest.fixture(scope="module")
def predictor_real():
    p = Predictor()
    p.cargar()
    return p


def test_verificacion_de_integridad(predictor_real):
    info = predictor_real.info()
    assert info["cargado"] is True
    assert info["hash_verificado"] is True
    assert info["version"] == "1.0.0"
    assert info["clases"] == ["negativo", "neutral", "positivo"]


def test_predicciones_coherentes(predictor_real):
    casos = {
        "El curso fue excelente y el profesor explica muy claro.": "positivo",
        "No me gustó: las clases fueron desorganizadas y sin retroalimentación.": "negativo",
        "El contenido estuvo bien, aunque faltaron ejemplos.": "neutral",
    }
    respuesta = predictor_real.predecir_lote(list(casos))
    for resultado, esperada in zip(respuesta["resultados"], casos.values()):
        assert resultado["polaridad"] == esperada
        assert abs(sum(resultado["probabilidades"].values()) - 1) < 1e-3
        assert 0 <= resultado["confianza"] <= 1


def test_texto_vacio(predictor_real):
    r = predictor_real.predecir_uno("")
    assert r["polaridad"] == "neutral"
    assert r["requiere_revision"] is True
    assert r["advertencia"] == "texto_vacio"


def test_latencia_por_comentario(predictor_real):
    respuesta = predictor_real.predecir_lote(["Excelente curso."] * 5)
    assert respuesta["latencia_ms"] >= 0
    assert respuesta["resumen"]["n"] == 5
