"""Núcleo de inferencia del MVP.

Carga una sola vez el Pipeline completo guardado con joblib (vectorizador de
embeddings + clasificador), verifica su integridad contra ``modelos/registro.json``
y expone predicciones con salida estructurada por comentario.

El módulo no conoce FastAPI ni Streamlit: ambas interfaces lo reutilizan.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import sklearn

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:  # necesario para deserializar preprocesamiento/representaciones
    sys.path.insert(0, str(SRC))

RAIZ = SRC.parent
MODELOS = RAIZ / "modelos"
MODELO_DEFECTO = MODELOS / "modelo_polaridad.joblib"
METADATA_DEFECTO = MODELOS / "metadata.json"
REGISTRO_DEFECTO = MODELOS / "registro.json"

TAREA = "polaridad_global"
CLASES_ESPERADAS = ("negativo", "neutral", "positivo")
MAX_CARACTERES = int(os.getenv("MVP_MAX_CARACTERES", "5000"))
MAX_LOTE = int(os.getenv("MVP_MAX_LOTE", "1000"))
UMBRAL_CONFIANZA = float(os.getenv("MVP_UMBRAL_CONFIANZA", "0.50"))
UMBRAL_MARGEN = float(os.getenv("MVP_UMBRAL_MARGEN", "0.10"))
VERIFICAR_HASH = os.getenv("MVP_VERIFICAR_HASH", "1") not in {"0", "false", "False"}
# El joblib no guarda MPNet (ver EmbeddingsFrase.__getstate__): sin calentamiento,
# el codificador se descarga/carga en la primera predicción (~7 s).
CALENTAR = os.getenv("MVP_CALENTAR", "1") not in {"0", "false", "False"}


class ErrorModelo(RuntimeError):
    """El modelo no se pudo cargar o no supera la verificación de integridad."""


class ErrorEntrada(ValueError):
    """La entrada no cumple el contrato de la API."""


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def _leer_json(ruta: Path) -> dict:
    if not ruta.exists():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


class Predictor:
    """Predictor perezoso y reutilizable del modelo de polaridad."""

    def __init__(
        self,
        modelo_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        registro_path: str | Path | None = None,
        verificar_hash: bool = VERIFICAR_HASH,
        umbral_confianza: float = UMBRAL_CONFIANZA,
        umbral_margen: float = UMBRAL_MARGEN,
        calentar: bool = CALENTAR,
    ) -> None:
        modelo_path = modelo_path or os.getenv("MVP_MODEL_PATH")
        self.modelo_path = Path(modelo_path) if modelo_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_DEFECTO
        self.registro_path = Path(registro_path) if registro_path else REGISTRO_DEFECTO
        self.verificar_hash = verificar_hash
        self.umbral_confianza = umbral_confianza
        self.umbral_margen = umbral_margen
        self.calentar = calentar
        self._modelo = None
        self._metadata: dict = {}
        self._registro: dict = {}
        self._version = "desconocida"
        self._hash: str | None = None
        self._hash_verificado = False
        self._cargado_en: float | None = None
        self._segundos_carga: float | None = None

    # ------------------------------------------------------------- carga
    def cargar(self) -> "Predictor":
        if self._modelo is not None:
            return self
        t0 = time.perf_counter()

        ruta, version, hash_esperado = self._resolver_archivo()
        if not ruta.exists() or ruta.stat().st_size == 0:
            raise ErrorModelo(f"No existe un modelo utilizable en {ruta}")
        if self.verificar_hash and self.modelo_path is None and not hash_esperado:
            raise ErrorModelo(
                "No hay una versión activa verificable en registro.json. "
                "Use gestion_modelos.py o defina MVP_MODEL_PATH para un archivo local."
            )

        hash_real = sha256_archivo(ruta)
        if self.verificar_hash and hash_esperado and hash_real != hash_esperado:
            raise ErrorModelo(
                "El SHA-256 del modelo no coincide con registro.json. "
                "No se carga un joblib no verificado."
            )

        try:
            modelo = joblib.load(ruta)
        except Exception as exc:  # mensaje accionable para el operador
            raise ErrorModelo(
                f"No se pudo deserializar {ruta.name}: {exc}. "
                "Verifique que el archivo sea de confianza y que las versiones "
                "de scikit-learn/sentence-transformers coincidan con las del entrenamiento."
            ) from exc

        if not hasattr(modelo, "predict_proba"):
            raise ErrorModelo("El Pipeline cargado no expone predict_proba.")
        clases = tuple(getattr(modelo, "classes_", ()))
        if clases and clases != CLASES_ESPERADAS:
            raise ErrorModelo(f"Clases inesperadas en el modelo: {clases}")

        if self.calentar:  # carga el codificador ahora y no en la primera petición
            try:
                modelo.predict_proba(["Comentario de calentamiento del modelo."])
            except Exception as exc:
                raise ErrorModelo(f"Falló el calentamiento del modelo: {exc}") from exc

        self._modelo = modelo
        self._metadata = _leer_json(self.metadata_path)
        self._registro = _leer_json(self.registro_path)
        self.modelo_path = ruta
        self._version = version
        self._hash = hash_real
        self._hash_verificado = bool(hash_esperado) and hash_real == hash_esperado
        self._cargado_en = time.time()
        self._segundos_carga = round(time.perf_counter() - t0, 2)
        return self

    def _resolver_archivo(self) -> tuple[Path, str, str | None]:
        registro = _leer_json(self.registro_path)
        activa = registro.get("version_activa")
        entradas = {m.get("version"): m for m in registro.get("modelos", [])}
        entrada = entradas.get(activa, {})

        if self.modelo_path is not None:
            ruta = self.modelo_path
        elif entrada.get("archivo"):
            ruta = self.registro_path.parent / entrada["archivo"]
        else:
            ruta = MODELO_DEFECTO

        version = str(activa or entrada.get("version") or "local")
        if self.modelo_path is not None and not entrada:
            version = "local"
        hash_esperado = entrada.get("sha256") if self.modelo_path is None else None
        return ruta, version, hash_esperado

    # ------------------------------------------------------------- info
    @property
    def cargado(self) -> bool:
        return self._modelo is not None

    def info(self) -> dict:
        meta = self._metadata or _leer_json(self.metadata_path)
        sklearn_entreno = meta.get("sklearn")
        return {
            "cargado": self.cargado,
            "version": self._version,
            "archivo": self.modelo_path.name if self.modelo_path else None,
            "sha256": self._hash,
            "hash_verificado": self._hash_verificado,
            "tarea": TAREA,
            "clases": list(meta.get("clases") or CLASES_ESPERADAS),
            "fecha_entrenamiento": meta.get("fecha"),
            "representacion": meta.get("rep"),
            "clasificador": meta.get("clf"),
            "metricas_test": {
                "accuracy": meta.get("test_accuracy"),
                "f1_macro": meta.get("test_f1_macro"),
                "f1_ic95": meta.get("test_f1_ic95"),
            },
            "matriz_confusion": meta.get("matriz_confusion"),
            "entorno": {
                "python_entrenamiento": meta.get("python"),
                "python_actual": platform.python_version(),
                "sklearn_entrenamiento": sklearn_entreno,
                "sklearn_actual": sklearn.__version__,
                "compatible": sklearn_entreno in (None, sklearn.__version__),
            },
            "umbrales": {
                "confianza": self.umbral_confianza,
                "margen": self.umbral_margen,
            },
            "segundos_carga": self._segundos_carga,
            "cargado_en": self._cargado_en,
        }

    # -------------------------------------------------------- predicción
    def predecir_uno(
        self,
        comentario: object,
        aspecto: str = "general",
        id_comentario: str | int | None = None,
        incluir_texto: bool = True,
    ) -> dict:
        respuesta = self.predecir_lote(
            [comentario],
            aspectos=[aspecto],
            ids=[id_comentario],
            incluir_texto=incluir_texto,
        )
        return respuesta["resultados"][0]

    def predecir_lote(
        self,
        comentarios: Sequence[object],
        aspectos: Sequence[str] | None = None,
        ids: Sequence[str | int | None] | None = None,
        incluir_texto: bool = True,
    ) -> dict:
        if not self.cargado:
            self.cargar()
        if isinstance(comentarios, (str, bytes)) or not isinstance(comentarios, Sequence):
            raise ErrorEntrada("'comentarios' debe ser una lista de textos.")
        if not comentarios:
            raise ErrorEntrada("La lista de comentarios está vacía.")
        if len(comentarios) > MAX_LOTE:
            raise ErrorEntrada(f"El lote supera el máximo de {MAX_LOTE} comentarios.")

        aspectos = list(aspectos) if aspectos is not None else ["general"] * len(comentarios)
        ids = list(ids) if ids is not None else list(range(1, len(comentarios) + 1))
        if len(aspectos) != len(comentarios) or len(ids) != len(comentarios):
            raise ErrorEntrada("'comentarios', 'aspectos' e 'ids' deben tener igual longitud.")

        limpios, avisos = [], []
        for valor in comentarios:
            texto, advertencia = self._limpiar_entrada(valor)
            limpios.append(texto)
            avisos.append(advertencia)

        t0 = time.perf_counter()
        try:
            probabilidades = self._modelo.predict_proba(limpios)
        except Exception as exc:
            raise ErrorModelo(f"Falló la inferencia: {exc}") from exc
        latencia_ms = round((time.perf_counter() - t0) * 1000, 2)

        clases = list(getattr(self._modelo, "classes_", CLASES_ESPERADAS))
        resultados = []
        for i, (texto, probs, aviso) in enumerate(zip(limpios, probabilidades, avisos)):
            resultados.append(
                self._estructurar(
                    texto=texto,
                    probs=probs,
                    clases=clases,
                    aspecto=str(aspectos[i] or "general"),
                    id_comentario=ids[i],
                    advertencia=aviso,
                    incluir_texto=incluir_texto,
                )
            )

        distribucion = {c: 0 for c in CLASES_ESPERADAS}
        for r in resultados:
            distribucion[r["polaridad"]] = distribucion.get(r["polaridad"], 0) + 1
        confianzas = [r["confianza"] for r in resultados]
        resumen = {
            "n": len(resultados),
            "distribucion": distribucion,
            "confianza_media": round(sum(confianzas) / len(confianzas), 4),
            "n_requiere_revision": sum(r["requiere_revision"] for r in resultados),
        }
        return {
            "version_modelo": self._version,
            "sha256_modelo": self._hash,
            "tarea": TAREA,
            "resultados": resultados,
            "resumen": resumen,
            "latencia_ms": latencia_ms,
        }

    # ----------------------------------------------------------- helpers
    def _limpiar_entrada(self, valor: object) -> tuple[str, str | None]:
        if valor is None or (isinstance(valor, float) and valor != valor):
            return "", "texto_vacio"
        if not isinstance(valor, str):
            valor = str(valor)
        texto = valor.strip()
        if not texto:
            return "", "texto_vacio"
        if len(texto) > MAX_CARACTERES:
            return texto[:MAX_CARACTERES], "texto_truncado"
        if len(texto) < 3:
            return texto, "texto_muy_corto"
        return texto, None

    def _estructurar(
        self,
        texto: str,
        probs: Iterable[float],
        clases: list[str],
        aspecto: str,
        id_comentario: str | int | None,
        advertencia: str | None,
        incluir_texto: bool,
    ) -> dict:
        probs = [float(p) for p in probs]
        orden = sorted(zip(clases, probs), key=lambda par: par[1], reverse=True)
        polaridad, confianza = orden[0]
        margen = confianza - orden[1][1] if len(orden) > 1 else confianza
        motivos = []
        if advertencia:
            motivos.append(advertencia)
        if confianza < self.umbral_confianza:
            motivos.append("baja_confianza")
        if margen < self.umbral_margen:
            motivos.append("clases_ambiguas")

        resultado = {
            "id_comentario": id_comentario,
            "aspecto": aspecto,
            "polaridad": polaridad,
            "confianza": round(confianza, 4),
            "margen": round(margen, 4),
            "probabilidades": {c: round(p, 4) for c, p in zip(clases, probs)},
            "requiere_revision": bool(motivos),
            "motivos_revision": motivos,
            "advertencia": advertencia,
        }
        if incluir_texto:
            resultado["comentario"] = texto
        return resultado
