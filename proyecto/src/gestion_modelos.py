"""Registro, activación y verificación de versiones del modelo.

Uso:
    python proyecto/src/gestion_modelos.py listar
    python proyecto/src/gestion_modelos.py verificar [--cargar]
    python proyecto/src/gestion_modelos.py registrar ARCHIVO --version 1.1.0 --notas "..." [--activar] [--forzar]
    python proyecto/src/gestion_modelos.py activar 1.0.0

El registro (modelos/registro.json) fija qué archivo joblib usa el MVP y su
SHA-256. Para revertir, active una versión anterior: el predictor la cargará en
el siguiente arranque.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MODELOS = RAIZ / "modelos"
REGISTRO = MODELOS / "registro.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predictor import Predictor, sha256_archivo  # noqa: E402


def _leer() -> dict:
    if not REGISTRO.exists():
        return {"version_activa": None, "modelos": []}
    return json.loads(REGISTRO.read_text(encoding="utf-8"))


def _guardar(registro: dict) -> None:
    MODELOS.mkdir(parents=True, exist_ok=True)
    temporal = REGISTRO.with_suffix(".tmp")
    temporal.write_text(json.dumps(registro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporal.replace(REGISTRO)


def _buscar(registro: dict, version: str) -> dict | None:
    return next((m for m in registro.get("modelos", []) if m.get("version") == version), None)


def listar() -> int:
    registro = _leer()
    activa = registro.get("version_activa")
    if not registro.get("modelos"):
        print("El registro está vacío.")
        return 1
    print(f"Versión activa: {activa}\n")
    for m in registro["modelos"]:
        marca = "*" if m["version"] == activa else " "
        print(f"{marca} {m['version']:10s} {m.get('fecha', ''):12s} {m.get('archivo', ''):38s} {m.get('notas', '')}")
    return 0


def verificar(cargar: bool = False) -> int:
    registro = _leer()
    version = registro.get("version_activa")
    entrada = _buscar(registro, version)
    if not entrada:
        print(f"FALLO: la versión activa '{version}' no existe en el registro.")
        return 1
    ruta = MODELOS / entrada["archivo"]
    if not ruta.exists():
        print(f"FALLO: no existe {ruta}.")
        return 1
    real = sha256_archivo(ruta)
    esperado = entrada.get("sha256")
    print(f"Versión activa: {version}")
    print(f"Archivo: {ruta.name}")
    print(f"SHA-256 registrado: {esperado}")
    print(f"SHA-256 real:       {real}")
    if real != esperado:
        print("FALLO: el hash no coincide. No active este archivo en producción.")
        return 1
    print("Integridad: OK")
    if cargar:
        p = Predictor()
        p.cargar()
        info = p.info()
        print("Carga: OK")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0 if info["hash_verificado"] else 1
    return 0


def registrar(archivo: Path, version: str, notas: str, activar: bool, forzar: bool) -> int:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        print("FALLO: la versión debe tener formato semver, por ejemplo 1.1.0.")
        return 1
    if not archivo.exists() or archivo.suffix != ".joblib":
        print(f"FALLO: {archivo} no es un archivo .joblib existente.")
        return 1

    registro = _leer()
    if _buscar(registro, version) and not forzar:
        print(f"FALLO: la versión {version} ya existe. Use --forzar para reemplazarla.")
        return 1

    destino = MODELOS / f"modelo_polaridad_{version}.joblib"
    MODELOS.mkdir(parents=True, exist_ok=True)
    if archivo.resolve() != destino.resolve():
        shutil.copy2(archivo, destino)
    entrada = {
        "version": version,
        "archivo": destino.name,
        "sha256": sha256_archivo(destino),
        "fecha": date.today().isoformat(),
        "notas": notas,
    }
    registro["modelos"] = [m for m in registro.get("modelos", []) if m.get("version") != version]
    registro["modelos"].insert(0, entrada)
    if activar or not registro.get("version_activa"):
        registro["version_activa"] = version
    _guardar(registro)
    print(f"Registrada la versión {version} ({entrada['sha256']}).")
    if registro["version_activa"] == version:
        print("Es la versión activa: reinicie la API o la app para cargarla.")
    return 0


def activar(version: str) -> int:
    registro = _leer()
    entrada = _buscar(registro, version)
    if not entrada:
        print(f"FALLO: la versión {version} no existe. Use 'listar'.")
        return 1
    ruta = MODELOS / entrada["archivo"]
    if not ruta.exists():
        print(f"FALLO: falta el archivo {ruta}.")
        return 1
    if sha256_archivo(ruta) != entrada.get("sha256"):
        print("FALLO: el archivo no coincide con el hash registrado.")
        return 1
    registro["version_activa"] = version
    _guardar(registro)
    print(f"Versión activa: {version}. Reinicie la API o la app para cargarla.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gestión de versiones del modelo del MVP.")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("listar", help="Mostrar versiones registradas.")
    p_ver = sub.add_parser("verificar", help="Comprobar el hash de la versión activa.")
    p_ver.add_argument("--cargar", action="store_true", help="Además, deserializar el Pipeline.")
    p_reg = sub.add_parser("registrar", help="Copiar y registrar un nuevo joblib.")
    p_reg.add_argument("archivo", type=Path)
    p_reg.add_argument("--version", required=True)
    p_reg.add_argument("--notas", default="")
    p_reg.add_argument("--activar", action="store_true")
    p_reg.add_argument("--forzar", action="store_true")
    p_act = sub.add_parser("activar", help="Activar una versión registrada.")
    p_act.add_argument("version")

    args = parser.parse_args()
    if args.comando == "listar":
        return listar()
    if args.comando == "verificar":
        return verificar(args.cargar)
    if args.comando == "registrar":
        return registrar(args.archivo, args.version, args.notas, args.activar, args.forzar)
    return activar(args.version)


if __name__ == "__main__":
    raise SystemExit(main())
