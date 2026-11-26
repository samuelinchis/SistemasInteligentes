"""Contratos de entrada y salida de la API del MVP."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PeticionComentario(BaseModel):
    comentario: str = Field(
        ...,
        description="Comentario de evaluación. Se acepta texto vacío para probar la validación.",
        examples=["El curso fue excelente y el profesor explica muy claro."],
    )
    aspecto: str = Field(
        "general",
        max_length=80,
        description="Aspecto al que se atribuye la predicción. El modelo actual predice polaridad global.",
    )
    id_comentario: str | int | None = Field(None, description="Identificador opaco para trazabilidad.")


class PeticionLote(BaseModel):
    comentarios: list[str | None] = Field(
        ...,
        min_length=1,
        description=(
            "Lista de comentarios. Máximo configurable con MVP_MAX_LOTE (1000 por defecto). "
            "Un null se trata como texto vacío y se marca para revisión, igual que una celda vacía del CSV."
        ),
        examples=[["Excelente curso.", "No me gustaron las clases."]],
    )
    aspectos: list[str | None] | None = Field(
        None, description="Un aspecto por comentario (opcional; null equivale a 'general')."
    )
    ids: list[str | int | None] | None = Field(None, description="Un identificador por comentario (opcional).")
    incluir_texto: bool = Field(True, description="Incluir el comentario recibido en la respuesta.")

    @model_validator(mode="after")
    def longitudes_coherentes(self):
        n = len(self.comentarios)
        if self.aspectos is not None and len(self.aspectos) != n:
            raise ValueError("'aspectos' debe tener la misma cantidad de elementos que 'comentarios'.")
        if self.ids is not None and len(self.ids) != n:
            raise ValueError("'ids' debe tener la misma cantidad de elementos que 'comentarios'.")
        return self


class Prediccion(BaseModel):
    id_comentario: str | int | None = None
    aspecto: str = "general"
    polaridad: str = Field(..., examples=["positivo"])
    confianza: float = Field(..., ge=0, le=1)
    margen: float = Field(..., ge=0, le=1)
    probabilidades: dict[str, float]
    requiere_revision: bool
    motivos_revision: list[str] = Field(default_factory=list)
    advertencia: str | None = None
    comentario: str | None = None


class ResumenPredicciones(BaseModel):
    n: int
    distribucion: dict[str, int]
    confianza_media: float
    n_requiere_revision: int


class RespuestaPrediccion(BaseModel):
    version_modelo: str
    sha256_modelo: str | None = None
    tarea: str
    resultados: list[Prediccion]
    resumen: ResumenPredicciones
    latencia_ms: float


class RespuestaCsv(RespuestaPrediccion):
    archivo: str
    columna: str
    filas_csv: int


class Salud(BaseModel):
    estado: str
    modelo_cargado: bool
    version_modelo: str | None = None
    hash_verificado: bool = False
    sklearn_compatible: bool = True
    sklearn_actual: str | None = None
    modelo_entrenado_con: str | None = None
    uptime_segundos: float
    error: str | None = None


class RespuestaMetricas(BaseModel):
    peticiones: int
    predicciones: int
    errores: int
    latencia_media_ms: float | None = None
    latencia_p95_ms: float | None = None
    distribucion: dict[str, int]
    uptime_segundos: float
