"""Pruebas del núcleo de inferencia."""
from __future__ import annotations

import json

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

import predictor as modulo_predictor
from predictor import ErrorEntrada, ErrorModelo, Predictor, sha256_archivo


def test_prediccion_individual_estructurada(predictor_falso):
    r = predictor_falso.predecir_uno("El curso fue excelente y el profesor explica claro.", id_comentario="a1")
    assert r["polaridad"] == "positivo"
    assert r["id_comentario"] == "a1"
    assert r["aspecto"] == "general"
    assert abs(sum(r["probabilidades"].values()) - 1) < 1e-6
    assert 0 <= r["confianza"] <= 1
    assert 0 <= r["margen"] <= 1
    assert r["requiere_revision"] is False


def test_texto_vacio_devuelve_neutral_y_revision(predictor_falso):
    r = predictor_falso.predecir_uno("")
    assert r["polaridad"] == "neutral"
    assert r["advertencia"] == "texto_vacio"
    assert r["requiere_revision"] is True
    assert "texto_vacio" in r["motivos_revision"]


def test_texto_truncado(monkeypatch, predictor_falso):
    monkeypatch.setattr(modulo_predictor, "MAX_CARACTERES", 10)
    r = predictor_falso.predecir_uno("a" * 50)
    assert r["advertencia"] == "texto_truncado"
    assert len(r["comentario"]) == 10
    assert r["requiere_revision"] is True


def test_lote_resumen_y_longitudes(predictor_falso):
    respuesta = predictor_falso.predecir_lote(
        ["Excelente curso.", "No me gustó.", "El contenido es regular."],
        aspectos=["general", "profesor", "contenido"],
        ids=["1", "2", "3"],
    )
    assert respuesta["tarea"] == "polaridad_global"
    assert respuesta["resumen"]["n"] == 3
    assert sum(respuesta["resumen"]["distribucion"].values()) == 3
    assert [r["aspecto"] for r in respuesta["resultados"]] == ["general", "profesor", "contenido"]
    assert respuesta["latencia_ms"] >= 0


def test_rechaza_lote_vacio(predictor_falso):
    with pytest.raises(ErrorEntrada):
        predictor_falso.predecir_lote([])


def test_rechaza_listas_incoherentes(predictor_falso):
    with pytest.raises(ErrorEntrada):
        predictor_falso.predecir_lote(["uno", "dos"], aspectos=["solo uno"])


def test_rechaza_limite_de_lote(monkeypatch, predictor_falso):
    monkeypatch.setattr(modulo_predictor, "MAX_LOTE", 2)
    with pytest.raises(ErrorEntrada):
        predictor_falso.predecir_lote(["uno", "dos", "tres"])


def test_hash_invalido_impide_carga(tmp_path):
    x = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = ["negativo", "neutral", "positivo", "neutral"]
    modelo = RandomForestClassifier(n_estimators=2, random_state=42).fit(x, y)
    archivo = tmp_path / "modelo.joblib"
    joblib.dump(modelo, archivo)
    registro = tmp_path / "registro.json"
    registro.write_text(
        json.dumps(
            {
                "version_activa": "9.9.9",
                "modelos": [
                    {"version": "9.9.9", "archivo": "modelo.joblib", "sha256": "incorrecto"}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ErrorModelo, match="SHA-256"):
        Predictor(registro_path=registro, metadata_path=tmp_path / "meta.json").cargar()


def test_modelo_con_clases_inesperadas_no_queda_cargado(tmp_path):
    x = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = ["neutral", "positivo", "neutral", "positivo"]
    modelo = RandomForestClassifier(n_estimators=2, random_state=42).fit(x, y)
    archivo = tmp_path / "modelo.joblib"
    joblib.dump(modelo, archivo)
    registro = tmp_path / "registro.json"
    registro.write_text(
        json.dumps(
            {
                "version_activa": "9.9.9",
                "modelos": [
                    {"version": "9.9.9", "archivo": "modelo.joblib", "sha256": sha256_archivo(archivo)}
                ],
            }
        ),
        encoding="utf-8",
    )
    p = Predictor(registro_path=registro, metadata_path=tmp_path / "meta.json")
    with pytest.raises(ErrorModelo, match="Clases inesperadas"):
        p.cargar()
    assert p.cargado is False


def test_info_reporta_version_y_entorno(predictor_falso):
    info = predictor_falso.info()
    assert info["cargado"] is True
    assert info["version"] == "test-1.0"
    assert info["tarea"] == "polaridad_global"
    assert info["clases"] == ["negativo", "neutral", "positivo"]
    assert info["metricas_test"]["f1_macro"] == 0.68
    assert len(info["matriz_confusion"]) == 3
    assert info["entorno"]["compatible"] is True
