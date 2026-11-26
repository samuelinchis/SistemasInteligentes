# Clasificación de evaluaciones docentes · MVP

MVP que usa el modelo de polaridad entrenado en la fase anterior para convertir
comentarios abiertos de estudiantes en una salida accionable: clase predicha,
probabilidades, confianza y una señal de revisión humana.

El modelo final es un `Pipeline` de scikit-learn guardado con joblib:
embeddings multilingües MPNet + Random Forest. Predice **polaridad global**
(`negativo`, `neutral`, `positivo`); el campo `aspecto` queda en la salida para
una futura extensión multiaspecto.

## Arquitectura

```
                    ┌──────────────────────────────┐
  Streamlit  ──────▶│  src/predictor.py            │
  (individual/CSV)  │  · carga única del joblib    │────▶ modelos/modelo_polaridad.joblib
                    │  · valida y verifica SHA-256 │      (MPNet + Random Forest)
  FastAPI    ──────▶│  · salida estructurada       │
  (/predecir, CSV)  └──────────────────────────────┘
                                │
                                ▼
                    modelos/registro.json (versión activa + hash)
```

- `src/predictor.py`: núcleo compartido. No conoce FastAPI ni Streamlit.
- `api/main.py`: servicio HTTP con validación, CORS, logs y métricas.
- `app/streamlit_app.py`: interfaz para un comentario o un CSV.
- `modelos/registro.json`: versión activa y SHA-256 del joblib.
- `deploy/`: imagen Docker y compose para API + interfaz.

## Inicio rápido

```bash
# 1. Entorno (Windows: venv\Scripts\python.exe en lugar de venv/bin/python)
python -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install -r proyecto/requirements-mvp.txt

# 2. Verificar que el joblib carga y predice
venv/bin/python proyecto/scripts/validar_modelo.py

# 3. Interfaz Streamlit
venv/bin/python -m streamlit run proyecto/app/streamlit_app.py

# 4. API (en otra terminal)
venv/bin/python -m uvicorn proyecto.api.main:app --host 127.0.0.1 --port 8000
```

- Streamlit: http://localhost:8501
- API y Swagger: http://localhost:8000/docs
- La primera predicción descarga MPNet (~1 GB) en la caché de Hugging Face y
  tarda unos 20 s; después el modelo queda en memoria y cada comentario tarda
  decenas de milisegundos en CPU.
- El MVP no reentrena al iniciar. Solo carga el joblib de confianza.

## Ejecución con Docker (sin instalar Python)

La imagen incluye Python 3.13, scikit-learn 1.9.1, joblib, PyTorch CPU, FastAPI
y Streamlit. No necesita `venv` ni paquetes locales: solo Docker con Compose.
Ejecute los comandos desde la raíz del repositorio (sirven igual en PowerShell,
CMD, macOS y Linux).

```bash
# 1. Construir la imagen (la primera vez tarda varios minutos)
docker compose -f proyecto/deploy/docker-compose.yml build

# 2. Levantar API + Streamlit en segundo plano
docker compose -f proyecto/deploy/docker-compose.yml up -d

# 3. Ver estado y logs
docker compose -f proyecto/deploy/docker-compose.yml ps
docker compose -f proyecto/deploy/docker-compose.yml logs -f api
```

- Streamlit: http://localhost:8501
- API y Swagger: http://localhost:8000/docs
- La primera predicción descarga MPNet (~1 GB) en el volumen `hf_cache`, que se
  conserva entre reinicios. Para descargarlo antes del primer uso:

```bash
docker compose -f proyecto/deploy/docker-compose.yml run --rm tools \
  python proyecto/scripts/validar_modelo.py
```

### Pruebas y utilidades dentro del contenedor

El servicio `tools` usa la misma imagen, pero no abre puertos ni arranca
servidores. Comparte la caché de Hugging Face con la API y Streamlit.

```bash
# 21 pruebas unitarias y de API (sin MPNet)
docker compose -f proyecto/deploy/docker-compose.yml run --rm tools python -m pytest -q

# 4 pruebas de integración con el joblib y MPNet reales
docker compose -f proyecto/deploy/docker-compose.yml run --rm -e RUN_INTEGRATION=1 tools \
  python -m pytest proyecto/tests/test_integracion.py -q

# Validación end-to-end del modelo
docker compose -f proyecto/deploy/docker-compose.yml run --rm tools \
  python proyecto/scripts/validar_modelo.py

# Shell interactiva con todas las dependencias instaladas
docker compose -f proyecto/deploy/docker-compose.yml run --rm tools
```

### Detener y limpiar

```bash
docker compose -f proyecto/deploy/docker-compose.yml down
docker compose -f proyecto/deploy/docker-compose.yml down -v   # borra también la caché de MPNet
```

Si cambia el código, el modelo o `requirements`, reconstruya con
`docker compose -f proyecto/deploy/docker-compose.yml up --build -d`.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Información del servicio |
| GET | `/salud` | Estado, versión y compatibilidad de scikit-learn |
| GET | `/modelo` | Ficha, métricas de test y versiones |
| GET | `/metricas` | Peticiones, predicciones, latencia p95 y distribución |
| POST | `/predecir` | Un comentario |
| POST | `/predecir/lote` | Lista de comentarios (máximo 1000) |
| POST | `/predecir/csv` | CSV con columna de comentarios; JSON o CSV etiquetado |

Ejemplo individual:

```bash
curl -X POST http://127.0.0.1:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{"comentario":"El curso fue excelente y el profesor explica muy claro.","aspecto":"contenido"}'
```

```json
{
  "version_modelo": "1.0.0",
  "sha256_modelo": "224bb1d8…",
  "tarea": "polaridad_global",
  "resultados": [
    {
      "id_comentario": null,
      "aspecto": "contenido",
      "polaridad": "positivo",
      "confianza": 0.6633,
      "margen": 0.4267,
      "probabilidades": {"negativo": 0.1, "neutral": 0.2367, "positivo": 0.6633},
      "requiere_revision": false,
      "motivos_revision": [],
      "advertencia": null,
      "comentario": "El curso fue excelente y el profesor explica muy claro."
    }
  ],
  "resumen": {
    "n": 1,
    "distribucion": {"negativo": 0, "neutral": 0, "positivo": 1},
    "confianza_media": 0.6633,
    "n_requiere_revision": 0
  },
  "latencia_ms": 38.4
}
```

Lote y CSV:

```bash
curl -X POST http://127.0.0.1:8000/predecir/lote \
  -H "Content-Type: application/json" \
  -d '{"comentarios":["Excelente curso.","No me gustaron las clases."]}'

# JSON con resultados
curl -X POST "http://127.0.0.1:8000/predecir/csv" \
  -F "archivo=@comentarios.csv"

# CSV descargable con predicciones
curl -X POST "http://127.0.0.1:8000/predecir/csv?formato=csv" \
  -F "archivo=@comentarios.csv" -o predicciones.csv
```

Si el CSV no se llama `comentario`, `feedback` o `texto`, indique la columna:
`?columna=mi_columna`. Se aceptan coma y punto y coma, UTF-8 y Latin-1. Si el
archivo trae una columna `id` o `aspecto`, se conservan en la respuesta y en el
CSV descargado.

## Contrato de salida

Cada comentario produce siempre la misma estructura:

- `polaridad`: `negativo`, `neutral` o `positivo`;
- `probabilidades`: una probabilidad por clase (suman 1);
- `confianza`: probabilidad de la clase elegida;
- `margen`: diferencia con la segunda clase;
- `requiere_revision`: `true` si hay baja confianza, clases ambiguas o texto inválido;
- `motivos_revision`: `texto_vacio`, `texto_truncado`, `texto_muy_corto`,
  `baja_confianza`, `clases_ambiguas`;
- `aspecto`: `general` por defecto. El modelo actual no predice aspectos.

Un texto vacío devuelve `neutral` con probabilidades casi uniformes y
`requiere_revision=true`; el caso se prueba automáticamente.

## Configuración

Variables de entorno (todas opcionales):

| Variable | Defecto | Uso |
|---|---|---|
| `MVP_MODEL_PATH` | versión activa del registro | joblib alternativo (no se verifica hash) |
| `MVP_VERIFICAR_HASH` | `1` | comparar SHA-256 con `registro.json` |
| `MVP_CALENTAR` | `1` | cargar MPNet al iniciar (una predicción de prueba) para que la primera petición no tarde ~7 s |
| `MVP_MAX_CARACTERES` | `5000` | truncar comentarios largos |
| `MVP_MAX_LOTE` | `1000` | máximo de filas por petición |
| `MVP_UMBRAL_CONFIANZA` | `0.50` | umbral para marcar revisión |
| `MVP_UMBRAL_MARGEN` | `0.10` | margen mínimo entre primera y segunda clase |
| `MVP_MAX_BYTES_CSV` | `8388608` | tamaño máximo del CSV |
| `MVP_CORS_ORIGINS` | `http://localhost:8501` | orígenes permitidos, separados por coma |

## Pruebas

```bash
# Unitarias y de API (no usan MPNet)
venv/bin/python -m pytest

# Integración con el joblib y MPNet reales
RUN_INTEGRATION=1 venv/bin/python -m pytest proyecto/tests/test_integracion.py
```

En Windows CMD: `set RUN_INTEGRATION=1` antes de pytest.

## Versiones y reversión

`modelos/registro.json` guarda la versión activa y el hash de su archivo. El
predictor se niega a cargar un joblib cuyo SHA-256 no coincida.

```bash
python proyecto/src/gestion_modelos.py listar
python proyecto/src/gestion_modelos.py verificar --cargar

# Registrar una nueva versión entrenada y activarla
python proyecto/src/gestion_modelos.py registrar ruta/al/nuevo.joblib \
  --version 1.1.0 --notas "nuevo entrenamiento" --activar

# Revertir a la versión anterior y reiniciar el servicio
python proyecto/src/gestion_modelos.py activar 1.0.0
```

Mantenga las versiones de scikit-learn y sentence-transformers iguales a las
del entrenamiento (`requirements.txt`). `/salud` informa si no coinciden.

## Despliegue

Local: `uvicorn` para la API y `streamlit` para la interfaz, como en el inicio
rápido. Para ejecutar todo en contenedores sin instalar Python, use la sección
«Ejecución con Docker». En un servidor:

```bash
docker compose -f proyecto/deploy/docker-compose.yml up --build -d
docker compose -f proyecto/deploy/docker-compose.yml ps
```

- API en `:8000`, Streamlit en `:8501`; volumen compartido para MPNet.
- La primera predicción descarga el modelo en el volumen `hf_cache`. Para
  evitarlo, prepare ese volumen o ejecute una predicción de calentamiento
  después del despliegue.
- En producción, ponga la API detrás de TLS y autenticación, limite el tamaño
  de las peticiones y no la exponga directamente a Internet sin control de acceso.

## Monitoreo

- `GET /metricas`: peticiones, predicciones, errores 5xx, latencia media/p95 y
  distribución de clases predichas.
- Logs JSON a stdout con `request_id`, ruta, estado y duración. **No** incluyen
  el texto del comentario ni identificadores de estudiantes.
- Cada respuesta de la API trae `X-Request-ID` para correlacionar logs.
- Para detectar deriva, compare la distribución de clases de `/metricas` con la
  del entrenamiento y revise la proporción de casos con `requiere_revision`.

## Privacidad y seguridad

- Los comentarios se procesan localmente; la app no los escribe en disco ni los
  envía a servicios externos. Los logs no guardan texto.
- El MVP solo deserializa el joblib del registro verificado. La deserialización
  de joblib puede ejecutar código: no cargue archivos recibidos de terceros.
- Anonimice los comentarios antes de usar cualquier servicio externo o LLM.
- La salida es apoyo a la lectura docente, no una decisión automática. Los casos
  con `requiere_revision=true` deben revisarse a mano.

## Límites conocidos

- Polaridad global, no nueve aspectos; `aspecto` es un campo preparado.
- Dataset pequeño (617 comentarios) y creado para el ejercicio.
- El neutral concentra la mayoría de los errores y los comentarios mixtos son
  difíciles.
- Un intervalo de confianza amplio en test (macro-F1 0,681; IC 95 % 0,58–0,76)
  implica que la confianza del modelo no debe interpretarse como certeza.
