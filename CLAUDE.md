# CLAUDE.md — Proyecto Sistemas Inteligentes (clasificación de evaluaciones docentes)

## Pendientes
- [ ] **Comenzar el MVP** (diapositivas 5 y 6 de `docs/CL12_26102026.pdf`). Pausado a pedido del usuario el 2026-11-05 para retomarlo después; ver la sección «Notas para el MVP» más abajo.
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
- Entorno: Python 3.13, pandas 3, scikit-learn 1.9, matplotlib, seaborn, joblib, reportlab. (sentence-transformers no está instalado aún.)

## Estructura
```
docs/        enunciado (PDF) y template de la revista
proyecto/
  data/raw/          CSV originales (entregados por el profesor; comentarios creados para el ejercicio)
  data/processed/    dataset depurado, errores_test.csv y revision_etiquetas.csv
  src/depurar.py           limpieza (pasos 1-2)
  src/preprocesamiento.py  normalizar_basico / normalizar_marcado (emojis, negación, intensificadores)
  src/representaciones.py  EmbeddingsFrase: transformador sklearn sobre sentence-transformers (con caché)
  src/experimentos.py      pasos 3-7: división, CV, GridSearch, test, figuras, guarda modelo
  resultados/        CSV/JSON de métricas (sin textos) + log
  modelos/           modelo_polaridad.joblib + metadata.json
  requirements.txt   versiones exactas usadas para entrenar
paper/
  paper.html         fuente del paper (editar aquí)
  paper.pdf          entregable
  figuras/           PNG generados por experimentos.py
```

## Cómo reproducir
```
python proyecto/src/depurar.py
PYTHONPATH=proyecto/src python proyecto/src/experimentos.py      # ~5 min en CPU
msedge --headless=new --no-pdf-header-footer --print-to-pdf=paper\paper.pdf paper\paper.html
```
Ojo: en figuras, el archivo `fig5_terminos.png` es la Figura 4 del paper y `fig4_matriz_confusion.png` la Figura 5.

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
