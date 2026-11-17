FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/datos/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# torch CPU primero para no descargar las ruedas CUDA en el servidor
COPY proyecto/requirements.txt /tmp/requirements.txt
COPY proyecto/requirements-mvp.txt /tmp/requirements-mvp.txt
RUN pip install --upgrade pip \
    && pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirements-mvp.txt

COPY proyecto /app/proyecto
COPY pytest.ini /app/pytest.ini
COPY .streamlit /app/.streamlit

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/salud || exit 1

# La API es el proceso por defecto; el compose usa la misma imagen para Streamlit.
CMD ["uvicorn", "proyecto.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
