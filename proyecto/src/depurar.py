from pathlib import Path
import pandas as pd

# --- Rutas ---------------------------------------------------------------
# __file__ = este script (src/depurar.py); .parent.parent sube hasta proyecto/
RAIZ = Path(__file__).resolve().parent.parent
# Carpeta con los CSV originales (nunca se modifican)
CARPETA_RAW = RAIZ / "data" / "raw"
# Carpeta de salida para el dataset depurado
CARPETA_PROCESSED = RAIZ / "data" / "processed"
# Crea processed/ si no existe (parents=True crea también las carpetas padre)
CARPETA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Lista ordenada de los CSV que coinciden con el patrón (el * captura el 2 y el 3)
IT_RAW = sorted(CARPETA_RAW.glob("feedback_curso_universitario*.csv"))

# --- Paso 1: carga -------------------------------------------------------
procesado = []  # aquí se guarda un DataFrame por archivo

for csv in IT_RAW:
    # utf_8_sig elimina el BOM del inicio, para que la primera columna se llame "id"
    df = pd.read_csv(csv, encoding='utf_8_sig')
    # Guarda de qué archivo viene cada fila (trazabilidad)
    df["archivo_origen"] = csv.name
    # Muestra nombre, forma (filas, columnas) y las 3 primeras filas de cada archivo
    print(f"\n{csv.name} -> {df.shape}")
    print(df.head(3))
    # Añade el DataFrame a la lista
    procesado.append(df)

# Une los tres DataFrames en uno; ignore_index renumera las filas de 0 a n-1
df_total = pd.concat(procesado, ignore_index=True)
print(f"\nFilas leídas (con basura): {len(df_total)}")

# --- Paso 2: eliminar encabezados repetidos ------------------------------
# Máscara booleana: True en las filas cuyo "id" es literalmente el texto "id"
mascara_encabezado = df_total["id"] == "id"
# Cuenta cuántas filas son encabezados repetidos (True cuenta como 1)
n_encabezados = mascara_encabezado.sum()
print(f"Encabezados repetidos: {n_encabezados}")
# Muestra cuáles son y de qué archivo vienen, antes de borrar nada
print(df_total.loc[mascara_encabezado, ["id", "archivo_origen"]])

# Se queda solo con las filas que NO son encabezado (~ invierte la máscara)
df_total = df_total[~mascara_encabezado].copy()

# --- Paso 3: convertir id a numérico -------------------------------------
# errors="coerce" convierte lo no numérico en NaN en lugar de lanzar error
df_total["id"] = pd.to_numeric(df_total["id"], errors="coerce")
# Comprueba si quedó algún id inválido
print(f"ids no numéricos tras la conversión: {df_total['id'].isna().sum()}")
# Los ids reinician en cada archivo, así que se crea un id único global
df_total["id_global"] = range(1, len(df_total) + 1)

# --- Paso 4: revisión del esquema ----------------------------------------
# Tipos de dato, no-nulos y memoria por columna
df_total.info()
# Conteo de valores faltantes por columna
print("\nFaltantes por columna:\n", df_total.isna().sum())
# Valores únicos de semestre, para decidir si es numérico o categórico
print("\nValores de semestre:", df_total["semestre"].unique())

# --- Paso 5: duplicados --------------------------------------------------
# Duplicados exactos: todas las columnas iguales (excepto id e info de origen)
cols_contenido = ["curso", "feedback", "polaridad_basica", "grano_fino",
                  "emocion", "aspectos", "semestre"]
n_dup_exactos = df_total.duplicated(subset=cols_contenido).sum()
# Duplicados solo por texto del comentario (los que más afectan al modelo)
n_dup_texto = df_total.duplicated(subset=["feedback"]).sum()
print(f"\nDuplicados exactos: {n_dup_exactos}")
print(f"Duplicados por feedback: {n_dup_texto}")
# keep=False muestra TODAS las copias de cada comentario repetido, para compararlas
repetidos = df_total[df_total.duplicated(subset=["feedback"], keep=False)]
print(repetidos.sort_values("feedback")[["id_global", "archivo_origen", "feedback",
                                         "polaridad_basica", "grano_fino"]].head(20))

# --- Resumen para el informe ---------------------------------------------
print(f"\nFilas finales: {len(df_total)}")
print(df_total["archivo_origen"].value_counts())

# --- Paso 6: normalizar polaridad_basica ---------------------------------
# Quita corchetes y comillas (['positivo'] -> positivo), pasa a minúsculas y recorta espacios
df_total["polaridad_basica"] = (
    df_total["polaridad_basica"]
    .str.replace(r"[\[\]'\"]", "", regex=True)
    .str.lower()
    .str.strip()
)
# Unifica la variante femenina con la masculina
df_total["polaridad_basica"] = df_total["polaridad_basica"].replace({"positiva": "positivo"})

# Define las únicas clases válidas y localiza las filas que no encajan (p. ej. "posicion")
CLASES_VALIDAS = {"positivo", "neutral", "negativo"}
mascara_invalida = ~df_total["polaridad_basica"].isin(CLASES_VALIDAS)
print(f"\nFilas con polaridad inválida: {mascara_invalida.sum()}")
# Muestra esas filas para revisarlas a mano antes de descartarlas
print(df_total.loc[mascara_invalida, ["id_global", "archivo_origen", "polaridad_basica", "feedback"]])
# Las descarta (decisión revisable: también podrías corregir la etiqueta a mano)
df_total = df_total[~mascara_invalida].copy()

# --- Paso 7: normalizar curso y texto ------------------------------------
# Corrige el error de escritura detectado en el nombre del curso
df_total["curso"] = df_total["curso"].str.strip().replace({"Sistemas opertativos": "Sistemas operativos"})
# Quita espacios sobrantes al inicio y al final de cada comentario
df_total["feedback"] = df_total["feedback"].str.strip()

# --- Paso 8: verificación final y guardado -------------------------------
print("\nDistribución de clases (polaridad_basica):")
print(df_total["polaridad_basica"].value_counts())
print("\nCursos:")
print(df_total["curso"].value_counts())

# Guarda el dataset depurado en processed/ (index=False evita una columna extra de índice)
SALIDA = CARPETA_PROCESSED / "dataset_depurado.csv"
df_total.to_csv(SALIDA, index=False, encoding="utf-8-sig")
print(f"\nGuardado: {SALIDA} ({len(df_total)} filas)")