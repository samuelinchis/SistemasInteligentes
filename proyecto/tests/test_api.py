"""Pruebas de la API FastAPI con un modelo simulado."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from proyecto.api.main import crear_app


@pytest.fixture
def cliente(predictor_falso):
    with TestClient(crear_app(predictor=predictor_falso)) as c:
        yield c


def test_raiz_y_salud(cliente):
    assert cliente.get("/").status_code == 200
    salud = cliente.get("/salud")
    assert salud.status_code == 200
    cuerpo = salud.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["version_modelo"] == "test-1.0"


def test_modelo_y_metricas(cliente):
    info = cliente.get("/modelo")
    assert info.status_code == 200
    assert info.json()["tarea"] == "polaridad_global"
    metricas = cliente.get("/metricas")
    assert metricas.status_code == 200
    assert "peticiones" in metricas.json()


def test_predecir_individual(cliente):
    r = cliente.post(
        "/predecir",
        json={"comentario": "El curso fue excelente y claro.", "aspecto": "contenido", "id_comentario": 7},
    )
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["version_modelo"] == "test-1.0"
    assert cuerpo["resultados"][0]["polaridad"] == "positivo"
    assert cuerpo["resultados"][0]["aspecto"] == "contenido"
    assert cuerpo["resultados"][0]["id_comentario"] == 7
    assert 0 <= cuerpo["resultados"][0]["confianza"] <= 1


def test_predecir_texto_vacio(cliente):
    r = cliente.post("/predecir", json={"comentario": ""})
    assert r.status_code == 200
    resultado = r.json()["resultados"][0]
    assert resultado["polaridad"] == "neutral"
    assert resultado["requiere_revision"] is True
    assert resultado["advertencia"] == "texto_vacio"


def test_predecir_lote_y_validacion(cliente):
    r = cliente.post(
        "/predecir/lote",
        json={"comentarios": ["Excelente curso.", "No me gustó."], "ids": ["a", "b"]},
    )
    assert r.status_code == 200
    assert r.json()["resumen"]["n"] == 2
    malo = cliente.post("/predecir/lote", json={"comentarios": ["uno", "dos"], "aspectos": ["x"]})
    assert malo.status_code == 422


def test_predecir_lote_acepta_null(cliente):
    r = cliente.post(
        "/predecir/lote",
        json={"comentarios": ["Excelente curso.", None], "aspectos": ["docencia", None]},
    )
    assert r.status_code == 200
    excelente, vacio = r.json()["resultados"]
    assert excelente["polaridad"] == "positivo"
    assert excelente["aspecto"] == "docencia"
    assert vacio["advertencia"] == "texto_vacio"
    assert vacio["requiere_revision"] is True
    assert vacio["aspecto"] == "general"
    assert vacio["comentario"] == ""


def test_csv_json_y_descarga(cliente):
    csv = "id,comentario\n1,Excelente curso.\n2,No me gustó la clase.\n".encode("utf-8")
    r = cliente.post(
        "/predecir/csv",
        files={"archivo": ("comentarios.csv", csv, "text/csv")},
    )
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["columna"] == "comentario"
    assert cuerpo["filas_csv"] == 2
    assert len(cuerpo["resultados"]) == 2

    descarga = cliente.post(
        "/predecir/csv",
        params={"formato": "csv"},
        files={"archivo": ("comentarios.csv", csv, "text/csv")},
    )
    assert descarga.status_code == 200
    assert descarga.headers["content-type"].startswith("text/csv")
    texto = descarga.content.decode("utf-8-sig")
    assert "comentario,aspecto,polaridad,confianza" in texto
    assert texto.count("\n") == 3


def test_csv_punto_y_coma_y_columna_explicita(cliente):
    csv = "feedback;curso\nExcelente curso.;Mate\nNo me gustó.;Física\n".encode("utf-8")
    r = cliente.post(
        "/predecir/csv",
        params={"columna": "feedback"},
        files={"archivo": ("evaluaciones.csv", csv, "text/csv")},
    )
    assert r.status_code == 200
    assert r.json()["columna"] == "feedback"
    assert r.json()["filas_csv"] == 2


def test_csv_detecta_columna_de_texto_arbitraria(cliente):
    csv = "id,opinion\n1,Excelente curso.\n2,No me gustó la clase.\n".encode("utf-8")
    r = cliente.post(
        "/predecir/csv",
        files={"archivo": ("opiniones.csv", csv, "text/csv")},
    )
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["columna"] == "opinion"
    assert cuerpo["filas_csv"] == 2
    assert cuerpo["resultados"][0]["comentario"] == "Excelente curso."


def test_csv_respeta_columna_aspecto(cliente):
    csv = "comentario,aspecto\nExcelente curso.,contenido\nNo me gustó.,metodología\n".encode("utf-8")
    r = cliente.post(
        "/predecir/csv",
        files={"archivo": ("aspectos.csv", csv, "text/csv")},
    )
    assert r.status_code == 200
    assert [p["aspecto"] for p in r.json()["resultados"]] == ["contenido", "metodología"]


def test_csv_invalido(cliente):
    no_csv = cliente.post(
        "/predecir/csv",
        files={"archivo": ("datos.txt", b"comentario\nhola\n", "text/plain")},
    )
    assert no_csv.status_code == 400
    vacio = cliente.post(
        "/predecir/csv",
        files={"archivo": ("datos.csv", b"id,comentario\n", "text/csv")},
    )
    assert vacio.status_code == 400
    columna = cliente.post(
        "/predecir/csv",
        params={"columna": "inexistente"},
        files={"archivo": ("datos.csv", b"id,comentario\n1,hola\n", "text/csv")},
    )
    assert columna.status_code == 400


def test_metricas_registran_predicciones(cliente):
    cliente.post("/predecir", json={"comentario": "Excelente curso."})
    datos = cliente.get("/metricas").json()
    assert datos["predicciones"] >= 1
    assert datos["distribucion"].get("positivo", 0) >= 1
    assert datos["latencia_media_ms"] is not None
