"""Servicio de predicción del MVP (FastAPI).

Ejecutar desde la raíz del repositorio:

    uvicorn proyecto.api.main:app --host 127.0.0.1 --port 8000

Documentación interactiva: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "proyecto" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from predictor import MAX_LOTE, ErrorEntrada, ErrorModelo, Predictor  # noqa: E402
from proyecto.api.esquemas import (  # noqa: E402
    PeticionComentario,
    PeticionLote,
    RespuestaCsv,
    RespuestaMetricas,
    RespuestaPrediccion,
    Salud,
)

VERSION_API = "1.0.0"
MAX_BYTES_CSV = int(os.getenv("MVP_MAX_BYTES_CSV", str(8 * 1024 * 1024)))
ORIGENES = [o.strip() for o in os.getenv("MVP_CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()]


def _log(evento: str, **campos) -> None:
    registro = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evento": evento,
        **campos,
    }
    print(json.dumps(registro, ensure_ascii=True, default=str), flush=True)


class Metricas:
    """Contadores en memoria. No registran textos ni identificadores de estudiantes."""

    def __init__(self) -> None:
        self.inicio = time.time()
        self.peticiones = 0
        self.predicciones = 0
        self.errores = 0
        self.latencias: deque[float] = deque(maxlen=1000)
        self.distribucion: Counter[str] = Counter()

    def registrar_peticion(self, latencia_ms: float, estado: int) -> None:
        self.peticiones += 1
        self.latencias.append(latencia_ms)
        if estado >= 500:
            self.errores += 1

    def registrar_prediccion(self, respuesta: dict) -> None:
        self.predicciones += len(respuesta.get("resultados", []))
        for clase, n in respuesta.get("resumen", {}).get("distribucion", {}).items():
            self.distribucion[clase] += int(n)

    def resumen(self) -> dict:
        lat = sorted(self.latencias)
        p95 = lat[max(0, math.ceil(0.95 * len(lat)) - 1)] if lat else None
        return {
            "peticiones": self.peticiones,
            "predicciones": self.predicciones,
            "errores": self.errores,
            "latencia_media_ms": round(sum(lat) / len(lat), 2) if lat else None,
            "latencia_p95_ms": round(p95, 2) if p95 is not None else None,
            "distribucion": dict(self.distribucion),
            "uptime_segundos": round(time.time() - self.inicio, 1),
        }


def _predictor_de(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    error = getattr(request.app.state, "error_modelo", None)
    if predictor is None or not predictor.cargado or error:
        raise HTTPException(status_code=503, detail=error or "Modelo no cargado.")
    return predictor


PredictorDep = Annotated[Predictor, Depends(_predictor_de)]


def _envelope(respuesta: dict) -> RespuestaPrediccion:
    return RespuestaPrediccion(**respuesta)


def _error_json(exc: Exception, codigo: int) -> JSONResponse:
    return JSONResponse(status_code=codigo, content={"detail": str(exc), "tipo": exc.__class__.__name__})


def _leer_tabla(contenido: bytes) -> pd.DataFrame:
    try:
        try:
            return pd.read_csv(io.BytesIO(contenido), encoding="utf-8-sig", sep=None, engine="python")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(contenido), encoding="latin-1", sep=None, engine="python")
    except Exception as exc:
        raise ErrorEntrada(f"No se pudo leer el CSV: {exc}") from exc


def _columna_texto(df: pd.DataFrame, solicitada: str | None) -> str:
    if solicitada:
        if solicitada not in df.columns:
            raise ErrorEntrada(
                f"La columna '{solicitada}' no existe. Columnas disponibles: {list(df.columns)}"
            )
        return solicitada

    por_nombre = {str(c).strip().casefold(): c for c in df.columns}
    for candidata in ("comentario", "feedback", "texto", "comment", "observacion", "observación"):
        if candidata in por_nombre:
            return str(por_nombre[candidata])

    for columna in df.columns:
        serie = df[columna]
        if pd.api.types.is_string_dtype(serie) and serie.astype(str).str.strip().ne("").any():
            return str(columna)
    raise ErrorEntrada(
        "No se detectó una columna de texto. Indique '?columna=...'. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def _resultados_a_csv(respuesta: dict) -> bytes:
    filas = []
    for i, r in enumerate(respuesta["resultados"], start=1):
        fila = {
            "fila": i,
            "id_comentario": r.get("id_comentario"),
            "comentario": r.get("comentario"),
            "aspecto": r.get("aspecto"),
            "polaridad": r.get("polaridad"),
            "confianza": r.get("confianza"),
            "margen": r.get("margen"),
            "requiere_revision": r.get("requiere_revision"),
            "motivos_revision": ";".join(r.get("motivos_revision") or []),
        }
        for clase, prob in (r.get("probabilidades") or {}).items():
            fila[f"prob_{clase}"] = prob
        filas.append(fila)
    return pd.DataFrame(filas).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def crear_app(predictor: Predictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.metricas = Metricas()
        app.state.error_modelo = None
        p = predictor or Predictor()
        try:
            if not p.cargado:
                p.cargar()
            app.state.predictor = p
            info = p.info()
            _log(
                "modelo_cargado",
                version=info["version"],
                archivo=info["archivo"],
                hash_verificado=info["hash_verificado"],
                sklearn_compatible=info["entorno"]["compatible"],
                segundos=info["segundos_carga"],
            )
        except Exception as exc:
            app.state.predictor = p
            app.state.error_modelo = str(exc)
            _log("error_carga_modelo", error=str(exc))
        yield

    app = FastAPI(
        title="Student Evaluation of Teaching · MVP",
        version=VERSION_API,
        description=(
            "API para clasificar comentarios de evaluación docente en negativo, neutral o positivo. "
            "Carga un Pipeline de scikit-learn + MPNet guardado con joblib y devuelve probabilidades, "
            "confianza y una señal de revisión humana. El modelo predice polaridad global; "
            "la salida se estructura por comentario y aspecto para permitir una futura extensión multiaspecto."
        ),
        lifespan=lifespan,
    )
    app.state.predictor = predictor
    app.state.metricas = Metricas()
    app.state.error_modelo = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGENES,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def trazabilidad(request: Request, call_next):
        t0 = time.perf_counter()
        request_id = (request.headers.get("x-request-id") or uuid.uuid4().hex)[:32]
        estado = 500
        try:
            respuesta = await call_next(request)
            estado = respuesta.status_code
            return respuesta
        finally:
            ms = round((time.perf_counter() - t0) * 1000, 2)
            request.app.state.metricas.registrar_peticion(ms, estado)
            if request.url.path not in {"/salud", "/metricas"}:
                _log(
                    "peticion",
                    request_id=request_id,
                    metodo=request.method,
                    ruta=request.url.path,
                    estado=estado,
                    ms=ms,
                )

    @app.exception_handler(ErrorEntrada)
    async def entrada_invalida(_: Request, exc: ErrorEntrada):
        return _error_json(exc, 400)

    @app.exception_handler(ErrorModelo)
    async def modelo_invalido(_: Request, exc: ErrorModelo):
        return _error_json(exc, 503)

    @app.get("/", tags=["metadatos"], summary="Información del servicio")
    def raiz():
        return {
            "servicio": "SET classification MVP",
            "version_api": VERSION_API,
            "tarea": "polaridad_global",
            "documentacion": "/docs",
            "endpoints": ["/salud", "/modelo", "/metricas", "/predecir", "/predecir/lote", "/predecir/csv"],
        }

    @app.get("/salud", response_model=Salud, tags=["metadatos"], summary="Estado y disponibilidad")
    def salud(response: Response, request: Request):
        p: Predictor | None = getattr(request.app.state, "predictor", None)
        error = getattr(request.app.state, "error_modelo", None)
        info = p.info() if p is not None else {}
        cargado = bool(p is not None and p.cargado)
        ok = cargado and not error
        if not ok:
            response.status_code = 503
        return Salud(
            estado="ok" if ok else "degradado",
            modelo_cargado=cargado,
            version_modelo=info.get("version"),
            hash_verificado=bool(info.get("hash_verificado")),
            sklearn_compatible=bool(info.get("entorno", {}).get("compatible", True)),
            sklearn_actual=info.get("entorno", {}).get("sklearn_actual"),
            modelo_entrenado_con=info.get("entorno", {}).get("sklearn_entrenamiento"),
            uptime_segundos=round(time.time() - request.app.state.metricas.inicio, 1),
            error=error,
        )

    @app.get("/modelo", tags=["metadatos"], summary="Ficha, métricas y versiones del modelo")
    def modelo(request: Request):
        p: Predictor | None = getattr(request.app.state, "predictor", None)
        if p is None:
            raise HTTPException(status_code=503, detail="Predictor no inicializado.")
        if getattr(request.app.state, "error_modelo", None):
            raise HTTPException(status_code=503, detail=request.app.state.error_modelo)
        return p.info()

    @app.get("/metricas", response_model=RespuestaMetricas, tags=["metadatos"], summary="Métricas de operación")
    def metricas(request: Request):
        return request.app.state.metricas.resumen()

    @app.post(
        "/predecir",
        response_model=RespuestaPrediccion,
        tags=["predicción"],
        summary="Clasificar un comentario",
    )
    def predecir(peticion: PeticionComentario, request: Request, predictor: PredictorDep):
        respuesta = predictor.predecir_lote(
            [peticion.comentario],
            aspectos=[peticion.aspecto],
            ids=[peticion.id_comentario],
            incluir_texto=True,
        )
        request.app.state.metricas.registrar_prediccion(respuesta)
        _log(
            "prediccion",
            n=1,
            latencia_ms=respuesta["latencia_ms"],
            revision=respuesta["resumen"]["n_requiere_revision"],
        )
        return _envelope(respuesta)

    @app.post(
        "/predecir/lote",
        response_model=RespuestaPrediccion,
        tags=["predicción"],
        summary="Clasificar una lista de comentarios",
    )
    def predecir_lote(peticion: PeticionLote, request: Request, predictor: PredictorDep):
        if len(peticion.comentarios) > MAX_LOTE:
            raise ErrorEntrada(f"El lote supera el máximo de {MAX_LOTE} comentarios.")
        respuesta = predictor.predecir_lote(
            peticion.comentarios,
            aspectos=peticion.aspectos,
            ids=peticion.ids,
            incluir_texto=peticion.incluir_texto,
        )
        request.app.state.metricas.registrar_prediccion(respuesta)
        _log(
            "prediccion_lote",
            n=len(peticion.comentarios),
            latencia_ms=respuesta["latencia_ms"],
            revision=respuesta["resumen"]["n_requiere_revision"],
        )
        return _envelope(respuesta)

    @app.post(
        "/predecir/csv",
        response_model=None,
        responses={200: {"model": RespuestaCsv}, 400: {"description": "CSV inválido"}},
        tags=["predicción"],
        summary="Clasificar un CSV y devolver JSON o CSV etiquetado",
    )
    def predecir_csv(
        request: Request,
        predictor: PredictorDep,
        archivo: Annotated[UploadFile, File(description="CSV con una columna de comentarios.")],
        columna: Annotated[str | None, Query(description="Nombre exacto de la columna de texto.")] = None,
        formato: Annotated[str, Query(pattern="^(json|csv)$")] = "json",
    ):
        if not archivo.filename or not archivo.filename.lower().endswith(".csv"):
            raise ErrorEntrada("El archivo debe tener extensión .csv.")
        contenido = archivo.file.read(MAX_BYTES_CSV + 1)
        if len(contenido) > MAX_BYTES_CSV:
            raise ErrorEntrada(f"El CSV supera el límite de {MAX_BYTES_CSV // (1024 * 1024)} MB.")
        if not contenido.strip():
            raise ErrorEntrada("El CSV está vacío.")

        df = _leer_tabla(contenido)
        if df.empty:
            raise ErrorEntrada("El CSV no contiene filas.")
        if len(df) > MAX_LOTE:
            raise ErrorEntrada(f"El CSV supera el máximo de {MAX_LOTE} filas.")
        col = _columna_texto(df, columna)
        textos = df[col].tolist()

        ids = None
        if "id" in {str(c).casefold() for c in df.columns}:
            col_id = next(c for c in df.columns if str(c).casefold() == "id")
            ids = [str(v) for v in df[col_id].tolist()]

        aspectos = None
        if "aspecto" in {str(c).casefold() for c in df.columns}:
            col_aspecto = next(c for c in df.columns if str(c).casefold() == "aspecto")
            aspectos = [
                str(v).strip() if pd.notna(v) and str(v).strip() else "general"
                for v in df[col_aspecto].tolist()
            ]

        respuesta = predictor.predecir_lote(textos, aspectos=aspectos, ids=ids, incluir_texto=True)
        request.app.state.metricas.registrar_prediccion(respuesta)
        _log(
            "prediccion_csv",
            archivo=archivo.filename,
            columna=col,
            filas=len(df),
            latencia_ms=respuesta["latencia_ms"],
            revision=respuesta["resumen"]["n_requiere_revision"],
        )

        if formato == "csv":
            nombre = f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            return StreamingResponse(
                iter([_resultados_a_csv(respuesta)]),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{nombre}"'},
            )
        return {
            **respuesta,
            "archivo": archivo.filename,
            "columna": col,
            "filas_csv": len(df),
        }

    return app


app = crear_app()
