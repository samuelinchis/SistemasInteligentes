"""Interfaz Streamlit del MVP de clasificación de evaluaciones docentes.

Ejecutar desde la raíz del repositorio:

    streamlit run proyecto/app/streamlit_app.py

La app carga el Pipeline una sola vez por proceso (st.cache_resource) y no
reentrena. Puede funcionar sin el servicio FastAPI; ambos comparten predictor.py.
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "proyecto" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predictor import ErrorEntrada, ErrorModelo, Predictor  # noqa: E402

AZUL = "#1b2a6b"
NARANJA = "#f2a51a"
COLORES = {"negativo": "#c0392b", "neutral": "#8a94a6", "positivo": "#2e86c1"}
ETIQUETAS = {"negativo": "Negativo", "neutral": "Neutral", "positivo": "Positivo"}
EJEMPLOS = {
    "Positivo": "El curso fue excelente. El profesor explica con claridad y los ejemplos ayudan a entender los temas.",
    "Negativo": "Las clases fueron desorganizadas y no hubo retroalimentación suficiente sobre los trabajos.",
    "Neutral": "El contenido estuvo bien, aunque faltaron ejemplos y el ritmo de las sesiones fue irregular.",
    "Ambiguo": "La metodología es interesante, pero la carga académica es excesiva y los criterios no son claros.",
}


st.set_page_config(
    page_title="Evaluación docente · MVP",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _css() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; }}
        .hero {{
            background: linear-gradient(110deg, {AZUL} 0%, #253f94 65%, #2e86c1 100%);
            border-radius: 14px; padding: 1.15rem 1.4rem; color: white;
            border-bottom: 5px solid {NARANJA}; margin-bottom: 1rem;
        }}
        .hero h1 {{ color: white; font-size: 1.65rem; margin: 0 0 .25rem 0; }}
        .hero p {{ color: #e8ecf8; margin: 0; font-size: .93rem; }}
        .badge {{
            display: inline-block; padding: .25rem .7rem; border-radius: 999px;
            color: white; font-weight: 700; font-size: .95rem;
        }}
        .card {{
            border: 1px solid #dfe4ef; border-radius: 12px; padding: .9rem 1.05rem;
            background: #fbfcff; margin-bottom: .8rem;
        }}
        .muted {{ color: #5f6b81; font-size: .85rem; }}
        .stButton>button {{ border-radius: 8px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Cargando modelo (solo la primera vez)…")
def cargar_predictor() -> Predictor:
    predictor = Predictor()
    predictor.cargar()
    return predictor


def _badge(clase: str) -> str:
    return (
        f'<span class="badge" style="background:{COLORES.get(clase, AZUL)}">'
        f"{ETIQUETAS.get(clase, clase)}</span>"
    )


def _mostrar_prediccion(r: dict, clave: str = "") -> None:
    clase = r["polaridad"]
    st.markdown(_badge(clase), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Confianza", f"{r['confianza']:.1%}")
    c2.metric("Margen sobre la 2.ª clase", f"{r['margen']:.1%}")
    c3.metric("Revisión humana", "Sí" if r["requiere_revision"] else "No")

    if r.get("advertencia") == "texto_vacio":
        st.warning("El comentario está vacío. El modelo devuelve una clase con muy poca información; se recomienda revisarlo.")
    elif r.get("advertencia") == "texto_truncado":
        st.warning("El comentario supera el máximo configurado y se truncó antes de predecir.")
    elif r.get("advertencia") == "texto_muy_corto":
        st.warning("El comentario es muy corto; la predicción tiene baja evidencia textual.")

    if r["requiere_revision"]:
        motivos = ", ".join(m.replace("_", " ") for m in r["motivos_revision"])
        st.info(f"Marcado para revisión humana: {motivos}.")

    probs = pd.DataFrame(
        {"Probabilidad": [r["probabilidades"].get(c, 0.0) for c in ETIQUETAS]},
        index=[ETIQUETAS[c] for c in ETIQUETAS],
    )
    st.bar_chart(probs, color=NARANJA, height=230)
    with st.expander("Salida estructurada (JSON)"):
        st.json(r, expanded=False)


def _leer_csv(contenido: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(contenido), encoding="utf-8-sig", sep=None, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(contenido), encoding="latin-1", sep=None, engine="python")


def _detectar_columna(df: pd.DataFrame) -> int:
    por_nombre = {str(c).strip().casefold(): i for i, c in enumerate(df.columns)}
    for candidata in ("comentario", "feedback", "texto", "comment", "observacion", "observación"):
        if candidata in por_nombre:
            return por_nombre[candidata]
    for i, columna in enumerate(df.columns):
        serie = df[columna]
        if pd.api.types.is_string_dtype(serie) and serie.astype(str).str.strip().ne("").any():
            return i
    return 0


def _tabla_descarga(respuesta: dict) -> bytes:
    filas = []
    for r in respuesta["resultados"]:
        fila = {
            "id_comentario": r.get("id_comentario"),
            "comentario": r.get("comentario"),
            "aspecto": r.get("aspecto"),
            "polaridad": r.get("polaridad"),
            "confianza": r.get("confianza"),
            "margen": r.get("margen"),
            "requiere_revision": r.get("requiere_revision"),
            "motivos_revision": ";".join(r.get("motivos_revision") or []),
        }
        for clase, prob in r["probabilidades"].items():
            fila[f"prob_{clase}"] = prob
        filas.append(fila)
    return pd.DataFrame(filas).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


_css()

try:
    predictor = cargar_predictor()
except (ErrorModelo, OSError) as exc:
    st.error(f"No se pudo cargar el modelo: {exc}")
    st.stop()

info = predictor.info()
metricas = info["metricas_test"]
f1_test = metricas.get("f1_macro") or 0.0
acc_test = metricas.get("accuracy") or 0.0
ic_test = metricas.get("f1_ic95") or []

st.markdown(
    f"""
    <div class="hero">
        <h1>Evaluación docente · clasificador de comentarios</h1>
        <p>MVP que convierte comentarios abiertos en polaridad accionable, con confianza y alertas
        para revisión humana. Modelo {info['version']} · macro-F1 en test {f1_test:.3f}.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Modelo en uso")
    st.markdown(f"**Versión** `{info['version']}`")
    st.caption(f"Archivo: `{info['archivo']}`")
    st.caption(f"SHA-256: `{(info['sha256'] or '')[:12]}…` · verificado: {'sí' if info['hash_verificado'] else 'no'}")
    st.metric("macro-F1 (test)", f"{f1_test:.3f}")
    st.metric("Exactitud (test)", f"{acc_test:.3f}")
    if len(ic_test) == 2:
        st.caption(f"IC 95 % macro-F1: [{ic_test[0]:.3f}; {ic_test[1]:.3f}]")
    if not info["entorno"]["compatible"]:
        st.warning(
            "La versión de scikit-learn actual no coincide con la del entrenamiento "
            f"({info['entorno']['sklearn_entrenamiento']})."
        )
    st.divider()
    st.markdown("### Criterio de revisión")
    st.caption(
        f"Se marca para revisión si la confianza es menor que {info['umbrales']['confianza']:.0%} "
        f"o si la diferencia con la segunda clase es menor que {info['umbrales']['margen']:.0%}."
    )
    st.divider()
    st.caption("Los comentarios se procesan localmente y no se guardan en disco ni se envían a servicios externos.")

tab_individual, tab_lote, tab_acerca = st.tabs(
    ["Comentario individual", "Carga por CSV", "Acerca del modelo"]
)

with tab_individual:
    st.subheader("Clasificar un comentario")
    st.caption("Ejemplos para probar:")
    columnas = st.columns(4)
    for columna, (nombre, ejemplo) in zip(columnas, EJEMPLOS.items()):
        if columna.button(nombre, width="stretch", key=f"ej_{nombre}"):
            st.session_state["texto_individual"] = ejemplo
            st.rerun()

    with st.form("form_individual"):
        texto = st.text_area(
            "Comentario de evaluación",
            height=130,
            placeholder="Escriba o pegue un comentario anónimo…",
            help="Evite incluir nombres, correos u otros datos personales.",
            key="texto_individual",
        )
        c1, c2 = st.columns([2, 1])
        aspecto = c1.text_input("Aspecto (opcional)", value="general")
        id_comentario = c2.text_input("Identificador (opcional)", value="")
        enviado = st.form_submit_button("Clasificar", type="primary", width="stretch")

    if enviado:
        try:
            st.session_state["resultado_individual"] = predictor.predecir_uno(
                texto,
                aspecto=(aspecto or "general")[:80],
                id_comentario=id_comentario or None,
            )
        except ErrorEntrada as exc:
            st.error(str(exc))

    if "resultado_individual" in st.session_state:
        st.divider()
        _mostrar_prediccion(st.session_state["resultado_individual"])

with tab_lote:
    st.subheader("Clasificar un archivo CSV")
    st.caption(
        "El archivo debe incluir una columna de comentarios (`comentario`, `feedback`, `texto` u otra que seleccione). "
        "Límite: 1000 filas por lote."
    )
    archivo = st.file_uploader("Archivo CSV", type=["csv"], accept_multiple_files=False)
    if archivo is not None:
        firma = f"{archivo.name}:{archivo.size}"
        if st.session_state.get("lote_firma") not in (None, firma):
            st.session_state.pop("lote", None)
        st.session_state["lote_firma"] = firma
        contenido = archivo.getvalue()
        try:
            df = _leer_csv(contenido)
        except Exception as exc:
            st.error(f"No se pudo leer el CSV: {exc}")
            df = None
        if df is not None and not df.empty:
            columna = st.selectbox("Columna de comentarios", list(df.columns), index=_detectar_columna(df))
            st.dataframe(df.head(8), width="stretch", hide_index=True)
            st.caption(f"{len(df):,} filas detectadas.")
            if len(df) > 1000:
                st.warning("El archivo supera 1000 filas. Se procesarán las primeras 1000.")
            if st.button("Clasificar archivo", type="primary"):
                try:
                    col_id = next((c for c in df.columns if str(c).casefold() == "id"), None)
                    ids = df[col_id].astype(str).tolist() if col_id is not None else None
                    aspectos = None
                    col_aspecto = next((c for c in df.columns if str(c).casefold() == "aspecto"), None)
                    if col_aspecto is not None:
                        aspectos = [
                            str(v).strip() if pd.notna(v) and str(v).strip() else "general"
                            for v in df[col_aspecto].head(1000).tolist()
                        ]
                    with st.spinner("Generando predicciones…"):
                        respuesta = predictor.predecir_lote(
                            df[columna].head(1000).tolist(),
                            aspectos=aspectos,
                            ids=ids[:1000] if ids else None,
                            incluir_texto=True,
                        )
                    st.session_state["lote"] = respuesta
                    st.session_state["lote_archivo"] = archivo.name
                except (ErrorEntrada, ErrorModelo) as exc:
                    st.error(str(exc))

        if "lote" in st.session_state:
            respuesta = st.session_state["lote"]
            resumen = respuesta["resumen"]
            st.divider()
            st.markdown(f"**Resultados de `{st.session_state.get('lote_archivo', 'CSV')}`**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Comentarios", resumen["n"])
            c2.metric("Positivos", resumen["distribucion"].get("positivo", 0))
            c3.metric("Neutrales", resumen["distribucion"].get("neutral", 0))
            c4.metric("Negativos", resumen["distribucion"].get("negativo", 0))

            dist = pd.DataFrame(
                {"Comentarios": [resumen["distribucion"].get(c, 0) for c in ETIQUETAS]},
                index=[ETIQUETAS[c] for c in ETIQUETAS],
            )
            st.bar_chart(dist, color=NARANJA, height=220)

            filas = []
            for r in respuesta["resultados"]:
                fila = {
                    "comentario": r.get("comentario"),
                    "aspecto": r.get("aspecto"),
                    "polaridad": r.get("polaridad"),
                    "confianza": r.get("confianza"),
                    "requiere_revision": r.get("requiere_revision"),
                    "motivos": ", ".join(r.get("motivos_revision") or []),
                }
                fila.update({f"p_{c}": r["probabilidades"].get(c) for c in ETIQUETAS})
                filas.append(fila)
            tabla = pd.DataFrame(filas)

            solo_revision = st.toggle("Mostrar solo casos que requieren revisión", value=False)
            vista = tabla[tabla["requiere_revision"]] if solo_revision else tabla
            st.dataframe(vista, width="stretch", hide_index=True)

            st.download_button(
                "Descargar predicciones (CSV)",
                data=_tabla_descarga(respuesta),
                file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary",
            )
            with st.expander("Resumen técnico"):
                st.json(
                    {
                        "version_modelo": respuesta["version_modelo"],
                        "latencia_ms": respuesta["latencia_ms"],
                        "confianza_media": resumen["confianza_media"],
                        "n_requiere_revision": resumen["n_requiere_revision"],
                    }
                )

with tab_acerca:
    st.subheader("Qué hace y qué no hace este MVP")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="card">
            <b>Modelo</b><br>
            Pipeline completo guardado con joblib: embeddings multilingües
            (<code>paraphrase-multilingual-mpnet-base-v2</code>) + Random Forest.
            Se cargó una vez y se reutiliza en cada predicción.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            **Salida por comentario**
            - polaridad (`negativo`, `neutral`, `positivo`);
            - probabilidades por clase y confianza;
            - margen frente a la segunda clase;
            - señal `requiere_revision` para el flujo humano;
            - aspecto declarado (por defecto `general`).
            """
        )
    with c2:
        st.markdown(
            """
            **Límites conocidos**
            - El modelo predice **polaridad global**, no nueve aspectos.
              El campo `aspecto` queda preparado para una extensión futura.
            - Los comentarios mixtos y los neutrales son los más difíciles:
              el neutral concentra la mayor parte de los errores.
            - La etiqueta automática no reemplaza la lectura del docente ni
              debe usarse para decisiones disciplinarias.
            - El dataset es pequeño (617 filas) y fue creado para el ejercicio.
            """
        )

    st.divider()
    st.markdown("#### Ficha técnica")
    izq, der = st.columns([1, 1])
    with izq:
        st.markdown(
            f"""
            - Versión del modelo: `{info['version']}`
            - Representación: `{info['representacion']}`
            - Clasificador: `{info['clasificador']}`
            - Fecha de entrenamiento: `{info['fecha_entrenamiento']}`
            - Python de entrenamiento: `{info['entorno']['python_entrenamiento']}`
            - scikit-learn de entrenamiento: `{info['entorno']['sklearn_entrenamiento']}`
            """
        )
        st.caption("Métricas en test (124 comentarios, 20 %):")
        st.dataframe(
            pd.DataFrame(
                {
                    "Métrica": ["Exactitud", "macro-F1"],
                    "Valor": [acc_test, f1_test],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with der:
        st.markdown("**Matriz de confusión (test)**")
        matriz = pd.DataFrame(
            info.get("matriz_confusion") or [],
            index=[f"real {c}" for c in ETIQUETAS],
            columns=[f"pred. {c}" for c in ETIQUETAS],
        )
        if not matriz.empty:
            st.dataframe(matriz, width="stretch")
        st.markdown(
            """
            **Servicio API**
            ```bash
            uvicorn proyecto.api.main:app --port 8000
            ```
            Endpoints: `/predecir`, `/predecir/lote`, `/predecir/csv`,
            `/salud`, `/modelo`, `/metricas` · documentación en `/docs`.
            """
        )

    with st.expander("Ver metadata completa del modelo"):
        st.json(info)
