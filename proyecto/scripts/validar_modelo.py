"""Validación de carga y predicciones del modelo del MVP.

Uso:
    python proyecto/scripts/validar_modelo.py

Comprueba integridad del joblib, coherencia de probabilidades, caso vacío y
tiempos de inferencia. Sale con código 1 si alguna verificación falla.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "proyecto" / "src"))

from predictor import Predictor  # noqa: E402

CASOS = [
    ("El curso fue excelente y el profesor explica muy claro.", "positivo"),
    ("No me gustó: las clases fueron desorganizadas y sin retroalimentación.", "negativo"),
    ("El contenido estuvo bien, aunque faltaron ejemplos.", "neutral"),
    ("La metodología es interesante, pero la carga académica es excesiva.", None),
]


def main() -> int:
    errores: list[str] = []
    print("Cargando y verificando el modelo…")
    t0 = time.perf_counter()
    predictor = Predictor()
    try:
        predictor.cargar()
    except Exception as exc:
        print(f"FALLO: no se pudo cargar el modelo: {exc}")
        return 1

    info = predictor.info()
    print(f"Carga de joblib: {round(time.perf_counter() - t0, 2)} s")
    print(json.dumps(info, indent=2, ensure_ascii=False))

    if not info["hash_verificado"]:
        errores.append("El SHA-256 no coincide con registro.json.")
    if not info["entorno"]["compatible"]:
        errores.append(
            "scikit-learn actual "
            f"{info['entorno']['sklearn_actual']} != entrenamiento "
            f"{info['entorno']['sklearn_entrenamiento']}."
        )

    t1 = time.perf_counter()
    respuesta = predictor.predecir_lote([texto for texto, _ in CASOS])
    primer_ms = (time.perf_counter() - t1) * 1000
    print(f"\nPrimera inferencia (incluye carga de MPNet): {primer_ms:.0f} ms")

    print("\nCasos de prueba:")
    for resultado, (_, esperada) in zip(respuesta["resultados"], CASOS):
        suma = sum(resultado["probabilidades"].values())
        print(
            f"  {resultado['polaridad']:8s} conf={resultado['confianza']:.3f} "
            f"esperada={esperada or 'libre':8s} {resultado['comentario'][:60]}"
        )
        if abs(suma - 1) > 1e-3:
            errores.append(f"Probabilidades no suman 1: {suma} en «{resultado['comentario'][:30]}».")
        if esperada and resultado["polaridad"] != esperada:
            errores.append(
                f"Se esperaba «{esperada}» y se obtuvo «{resultado['polaridad']}» "
                f"en «{resultado['comentario'][:40]}»."
            )

    vacio = predictor.predecir_uno("")
    print(f"\nTexto vacío -> {vacio['polaridad']} (conf={vacio['confianza']:.3f}, "
          f"revisión={vacio['requiere_revision']})")
    if vacio["polaridad"] != "neutral":
        errores.append("Un texto vacío debería devolver neutral.")
    if not vacio["requiere_revision"]:
        errores.append("Un texto vacío debería marcarse para revisión.")

    latencias = []
    for _ in range(3):
        t = time.perf_counter()
        predictor.predecir_lote([texto for texto, _ in CASOS])
        latencias.append((time.perf_counter() - t) * 1000)
    mediana = statistics.median(latencias)
    print(f"Latencia mediana del lote de {len(CASOS)} comentarios: {mediana:.0f} ms "
          f"({mediana / len(CASOS):.0f} ms por comentario)")

    resumen = {
        "ok": not errores,
        "errores": errores,
        "version_modelo": info["version"],
        "primer_comentario_ms": round(primer_ms, 1),
        "latencia_mediana_ms": round(mediana, 1),
        "latencia_por_comentario_ms": round(mediana / len(CASOS), 1),
    }
    print("\n" + json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0 if not errores else 1


if __name__ == "__main__":
    raise SystemExit(main())
