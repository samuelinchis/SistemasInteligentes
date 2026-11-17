# CLAUDE.md — Proyecto Sistemas Inteligentes (clasificación de evaluaciones docentes)

## Pendientes
- [x] **MVP implementado** (diapositivas 5 y 6 de `docs/CL12_26102026.pdf`): predictor compartido, API FastAPI, interfaz Streamlit, registro de versiones con verificación SHA-256, pruebas y Docker (2026-11-15). Ver «MVP» más abajo.
- [x] Paper (`paper/paper.pdf`) terminado, revisado y subido a `main` (2026-11-05).

## Contexto académico
- Curso: Sistemas Inteligentes. Equipo de cinco integrantes (ver «Estado»); remoto `samuelinchis/SistemasInteligentes`.
- Parte anterior (20 %, ya entregada, fuera de este repo): `../Informe_ML.pdf` + notebooks de ML supervisado/no supervisado con datos sintéticos.
- **Parte actual (20 %)**: investigación + **paper** sobre clasificación de comentarios de estudiantes (Student Evaluation of Teaching, SET). Entregable final en **PDF**.
- **Después**: un MVP básico que use el modelo entrenado (ver diapositivas 5 y 6).

## Enunciado (docs/CL12_26102026.pdf)
- Diap. 2: contexto (de lexicones a transformers/LLMs), reto (poca adopción y accionabilidad), objetivo (evaluar un modelo ML y luego integrarlo en un MVP).
- Diap. 3: herramientas sugeridas. Los LLM son asistentes metodológicos; etiquetas automáticas requieren validación humana; anonimizar comentarios antes de enviarlos a servicios externos.
- **Diap. 4 (lo que se evalúa en el paper)**:
  1. Carga y exploración del CSV (columnas, etiquetas, faltantes, duplicados, calidad).
  2. Definición de la tarea y preprocesamiento (normalizar texto, faltantes, emojis, intensificadores).
  3. División train 80 % / test 20 % estratificada, sin fuga de información, agrupando por autor.
  4. Representación: TF-IDF vs. embeddings semánticos (ajustados solo con train).
  5. Línea base: DummyClassifier; comparar Logistic Regression, Linear SVM, Naive Bayes, Random Forest.
  6. Hiperparámetros: Pipeline + GridSearchCV/RandomizedSearchCV, CV estratificada, selección por macro-F1.
  7. Evaluación en test: accuracy, precision, recall, macro-F1, matriz de confusión + análisis de errores.
  - Formato: template de la revista *Ingeniería y Universidad* (Javeriana): https://revistas.javeriana.edu.co/index.php/iyu/about/submissions
- Diap. 5–6 (MVP, fase posterior): guardar Pipeline completo con joblib, validar carga, API FastAPI/Flask, interfaz Streamlit (comentario individual o CSV), versionado, salida estructurada por aspecto.

## Template del paper (docs/Template Submission-OTH.docx)
Título en inglés y en español · Abstract estructurado ≤200 palabras (objetivos, materiales y métodos, resultados y discusión, conclusiones) + Keywords (≥3) · Resumen + Palabras clave · Introduction · Materials and methods · Results · Conclusions · References (**formato IEEE**, citas numéricas [1]).
Cada tabla/figura: título arriba ("Table 1. …") y fuente abajo ("Source: Authors own creation.").

## Decisiones tomadas (2026-11-05)
- Cuerpo del paper en **español**; título y abstract también en inglés (exigido por el template).
- Estética: **estructura exacta del template** (una columna, sin portada) con la **paleta del Informe_ML** en títulos y tablas.
- Tarea: **solo polaridad** (positivo / neutral / negativo). Aspectos queda como trabajo futuro.
- Embeddings: sentence-transformers con modelo multilingüe pequeño, ejecutado localmente.

## Datos
- Originales en `proyecto/data/raw/feedback_curso_universitario{,2,3}.csv`. Son comentarios creados por el profesor para el ejercicio; todo el dataset está versionado en git.
- Columnas: `id, curso, feedback, polaridad_basica, grano_fino, emocion, aspectos, semestre`.
- Hallazgos de la exploración:
  - 618 filas útiles (617 tras depurar); 7 cursos; semestres 2023-1 a 2024-2.
  - Cada CSV repite la fila de encabezado; los `id` reinician por archivo.
  - `polaridad_basica` con formatos mezclados (`neutral`, `['neutral']`, `positiva`, `["['positivo']"]`, `['posicion']`). Tras normalizar: neutral 259, negativo 205, positivo 153.
  - `grano_fino` casi constante (612 de 618 = 3) y `emocion` casi constante (610 = neutral): **no sirven como objetivo** de clasificación.
  - `aspectos`: lista JSON de `{aspecto, sentimiento}` con nombres no normalizados (mayúsculas, sinónimos) → posible tarea multi-etiqueta secundaria.
  - No hay columna de autor → "agrupar por autor" no es posible directamente; se puede aproximar con detección de casi-duplicados.
  - Ruido de etiquetas visible (p. ej. comentarios claramente positivos marcados `negativo`) y textos que no son comentarios (p. ej. "Estimado estudiante, …").
  - Error tipográfico en curso: `Sistemas opertativos` → `Sistemas operativos`.

## Código
- `proyecto/src/depurar.py`: limpieza (pasos 1–2 de la diap. 4). Une los 3 CSV, quita encabezados repetidos, crea `id_global`, normaliza polaridad y curso, guarda `proyecto/data/processed/dataset_depurado.csv`. Ejecutar: `python proyecto/src/depurar.py`.
- `proyecto/src/preprocesamiento.py`: `normalizar_basico` / `normalizar_marcado` (emojis, negación, intensificadores).
- `proyecto/src/representaciones.py`: `EmbeddingsFrase`, transformador sklearn sobre sentence-transformers (con caché).
- `proyecto/src/experimentos.py`: pasos 3–7 del paper; guarda `modelo_polaridad.joblib` y `metadata.json`.
- `proyecto/src/predictor.py`: núcleo del MVP. Carga perezosa, verifica SHA-256 contra `modelos/registro.json`, valida entradas (vacío, truncado, lote), devuelve salida estructurada con probabilidades, confianza, margen y `requiere_revision`.
- `proyecto/src/gestion_modelos.py`: `listar` / `verificar` / `registrar` / `activar` versiones (reversión sin tocar código).
- `proyecto/api/main.py`: FastAPI. Endpoints `/predecir`, `/predecir/lote`, `/predecir/csv`, `/salud`, `/modelo`, `/metricas`; logs JSON sin textos y CORS configurable.
- `proyecto/app/streamlit_app.py`: interfaz de comentario individual y CSV, con descarga de predicciones y alertas de revisión.
- Entorno: Python 3.13 en entrenamiento; MVP probado localmente con Python 3.12 y scikit-learn 1.9.1. (sentence-transformers 6.2.0, torch 2.15.0+cpu.)

## Estructura
```
docs/        enunciado (PDF) y template de la revista
proyecto/
  data/raw/          CSV originales (entregados por el profesor; comentarios creados para el ejercicio)
  data/processed/    dataset depurado, errores_test.csv y revision_etiquetas.csv
  src/
    depurar.py               limpieza (pasos 1-2)
    preprocesamiento.py      normalizar_basico / normalizar_marcado
    representaciones.py      EmbeddingsFrase sobre sentence-transformers
    experimentos.py          pasos 3-7 del paper
    predictor.py             núcleo del MVP (carga, validación, salida estructurada)
    gestion_modelos.py       registro y reversión de versiones
  api/main.py                FastAPI
  app/streamlit_app.py       interfaz Streamlit
  scripts/validar_modelo.py  smoke test de carga, coherencia y latencia
  tests/                     pytest unitario, API e integración con MPNet
  deploy/                    Dockerfile + docker-compose (API + Streamlit)
  resultados/        CSV/JSON de métricas (sin textos) + log
  modelos/           modelo_polaridad.joblib + metadata.json + registro.json
  requirements.txt       versiones exactas usadas para entrenar
  requirements-mvp.txt   API, Streamlit y pruebas
paper/
  paper.html         fuente del paper (editar aquí)
  paper.pdf          entregable
  figuras/           PNG generados por experimentos.py
.streamlit/config.toml   tema de la interfaz
pytest.ini               configuración de pruebas
```

## Cómo reproducir
```
python proyecto/src/depurar.py
PYTHONPATH=proyecto/src python proyecto/src/experimentos.py      # ~5 min en CPU
msedge --headless=new --no-pdf-header-footer --print-to-pdf=paper\paper.pdf paper\paper.html
```
Ojo: en figuras, el archivo `fig5_terminos.png` es la Figura 4 del paper y `fig4_matriz_confusion.png` la Figura 5.

## MVP (2026-11-15)
- Entorno local usado: Python 3.12.4 + `venv/` (ignorado por git), scikit-learn 1.9.1, sentence-transformers 6.2.0, torch 2.15.0+cpu, FastAPI 0.163.1, Streamlit 1.67.0.
- Instalar: `pip install -r proyecto/requirements-mvp.txt` (torch CPU primero, ver `proyecto/README.md`).
- Validar: `python proyecto/scripts/validar_modelo.py` → joblib verificado, 3 casos correctos, vacío neutral con revisión, ~30 ms/comentario después de cargar MPNet.
- API: `python -m uvicorn proyecto.api.main:app --port 8000`; Streamlit: `python -m streamlit run proyecto/app/streamlit_app.py`.
- Pruebas: `python -m pytest` (21 pruebas sin MPNet) y `RUN_INTEGRATION=1 python -m pytest proyecto/tests/test_integracion.py` (4 pruebas reales). Todas pasan.
- Salida: `{polaridad, confianza, margen, probabilidades, aspecto, requiere_revision, motivos_revision}`. `aspecto` es un campo preparado; el modelo solo predice polaridad global.
- Versión activa: `modelo_polaridad.joblib` v1.0.0, SHA-256 `224bb1d8…`, en `modelos/registro.json`. Reversión con `gestion_modelos.py activar`.
- Privacidad: los logs no guardan comentarios; la app procesa localmente. No cargar joblib de terceros.
- Docker: `proyecto/deploy/Dockerfile` + `docker-compose.yml` (API 8000, Streamlit 8501, caché HF compartida, servicio `tools` para pruebas/scripts). Verificado: `up --build -d`, `/salud`, `/predecir` real, `pytest` (21) y 4 de integración dentro del contenedor. No requiere Python local.
- Resultados de la validación end-to-end: `/predecir` correcto; CSV de 617 filas procesado por la API (~134 s) con distribución neg 200 / neu 273 / pos 144 y 37 casos marcados para revisión. La app Streamlit se probó con `AppTest`: sin excepciones y predicción individual correcta.
- Cambios del MVP publicados en la rama `MVP_V1` (no en `main`), 2026-11-16.

## Estado (2026-11-05)
- Autores (en este orden): Nicolás Gómez Giraldo, Samuel Rueda Sánchez, José Harvey Barrera Ramos, Esteban Muñoz Serrano, Jhon David Rios Aguirre. Paper redactado en **impersonal** (varios autores).
- Revisión de etiquetas: `proyecto/data/processed/revision_etiquetas.csv` (juicio del LLM + columna `juicio_humano`). El usuario validó los 28 casos señalados (incoherentes/incorrectas, discutibles, errores del modelo, no evaluaciones); los «coherentes» y «mixtos» no se revisaron uno por uno.
- Textos que no son evaluaciones: ids 603, 607 (test) y 616 (train). Decisión: conservarlos y documentarlos (opción A).
- Revisión de trampas hecha: template, presentación y dataset sin texto oculto ni instrucciones dirigidas a IA.
- Paper terminado (11 páginas, 8 tablas, 6 figuras). Resultados clave: embeddings > TF-IDF (~0,72 vs ~0,63 macro-F1 en CV); modelo final MPNet + Random Forest, test macro-F1 0,681, exactitud 0,694, IC95 [0,58; 0,76]; Dummy 0,197.
- Hallazgos de datos: 83 % de comentarios con conectores de contraste; 25 % de los neutrales contienen la palabra «neutral»; textos que no son evaluaciones (generados automáticamente); auditoría LLM de 50 comentarios: 68 % coherentes, 24 % discutibles, 8 % incoherentes (pendiente validación humana).

## Notas para el MVP (siguiente fase)
- Cargar `proyecto/modelos/modelo_polaridad.joblib` con `proyecto/src` en el `sys.path` (el pipeline referencia `preprocesamiento` y `representaciones`). La primera carga tarda ~8 s (MPNet); luego ~18 ms por comentario. Tiene `predict_proba`.
- Validar entrada: un texto vacío devuelve «neutral» con probabilidades casi uniformes.
- Sugerido por la diap. 5-6: Streamlit (comentario individual + carga CSV) y, si se separa backend, FastAPI. No reentrenar al iniciar la app. Solo cargar joblib de confianza.

## Convenciones de redacción (preferencia del usuario)
- Español, primera persona del singular o impersonal; sin signos de exclamación ni diminutivos.
- Estética de referencia: `../Informe_ML.pdf` (azul marino #1a2a6c aprox., acentos naranja/ámbar, tablas con encabezado azul, figuras numeradas con pie en cursiva).
- No commitear sin que el usuario lo pida. El repo incluye todo el proyecto (datos, modelo, PDFs); el .gitignore solo excluye cachés, venv y .env.
