## Datos

El dataset fue entregado por el profesor del curso y está formado por comentarios de
estudiantes creados para el ejercicio (no son evaluaciones reales). Está incluido en el
repositorio:

- `data/raw/`: los tres CSV originales, sin modificar
  (`feedback_curso_universitario.csv`, `feedback_curso_universitario2.csv`,
  `feedback_curso_universitario3.csv`).
- `data/processed/dataset_depurado.csv`: dataset limpio que genera `src/depurar.py`.
- `data/processed/errores_test.csv`: comentarios de prueba mal clasificados por el modelo final.
- `data/processed/revision_etiquetas.csv`: auditoría de etiquetas (juicio del LLM y validación humana).

## Cómo ejecutar

```
pip install -r proyecto/requirements.txt
python proyecto/src/depurar.py
PYTHONPATH=proyecto/src python proyecto/src/experimentos.py
```

El modelo entrenado ya está en `modelos/modelo_polaridad.joblib`, así que no hace falta
reentrenar para usarlo. Más contexto en `CLAUDE.md`, en la raíz del repositorio.
