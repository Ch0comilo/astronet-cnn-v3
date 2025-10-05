import os
import sys
import json
import subprocess
import requests

# ----------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------
MODEL_DIR = "/workspace/exoplanet-ml/MODEL_DIR"
KEPLER_DATA_DIR = "/workspace/exoplanet-ml/KEPLER_DATA_DIR"
OUTPUT_DIR = "/workspace/exoplanet-ml/kepler_pictures"

API_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# ----------------------------
# FUNCIÓN PARA OBTENER DATOS DEL ARCHIVO NASA
# ----------------------------
def get_kepler_params(kepler_id: str):
    """Obtiene period, t0 y duration desde la API oficial del NASA Exoplanet Archive."""
    query = (
        f"SELECT koi.kepid, koi.koi_period, koi.koi_time0bk, koi.koi_duration "
        f"FROM q1_q17_dr25_koi koi "
        f"WHERE koi.kepid = {kepler_id}"
    )
    params = {"query": query, "format": "json"}

    print(f"🔍 Consultando parámetros para Kepler ID {kepler_id}...")
    response = requests.get(API_URL, params=params)
    response.raise_for_status()

    data = response.json()
    if not data:
        raise ValueError(f"No se encontraron datos para el Kepler ID {kepler_id}")

    entry = data[0]
    period = entry.get("koi_period")
    t0 = entry.get("koi_time0bk")
    duration = entry.get("koi_duration")

    if None in (period, t0, duration):
        raise ValueError(f"Datos incompletos para el Kepler ID {kepler_id}")

    print(f"✅ Parámetros obtenidos:")
    print(f"   Period  = {period} días")
    print(f"   T0      = {t0} BKJD")
    print(f"   Duration= {duration} horas")

    return period, t0, duration/24


# ----------------------------
# DESCARGAR ARCHIVOS FITS
# ----------------------------
def download_fits_files(kepler_id: str):
    """Descarga los archivos FITS para el ID indicado."""
    prefix = kepler_id[:4]
    save_dir = os.path.join(KEPLER_DATA_DIR, prefix, kepler_id)
    os.makedirs(save_dir, exist_ok=True)

    url = f"http://archive.stsci.edu/pub/kepler/lightcurves/{prefix}/{kepler_id}/"
    print(f"⬇️  Descargando FITS desde {url} ...")

    cmd = [
        "wget",
        "-nH", "--cut-dirs=6", "-r", "-l0", "-c", "-N", "-np",
        "-erobots=off", "-R", "index*", "-A", "*_llc.fits",
        "-P", save_dir, url
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Archivos FITS guardados en {save_dir}")
    return save_dir


# ----------------------------
# EJECUTAR EL MODELO ASTRONET
# ----------------------------
def run_astronet(kepler_id: str, period: float, t0: float, duration: float):
    """Ejecuta el modelo Astronet para el Kepler ID."""
    output_file = os.path.join(OUTPUT_DIR, f"{kepler_id}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        "python", "-m", "astronet.predict",
        "--model=AstroCNNModel",
        "--config_name=local_global",
        f"--model_dir={MODEL_DIR}",
        f"--kepler_data_dir={KEPLER_DATA_DIR}",
        f"--kepler_id={kepler_id}",
        f"--period={period}",
        f"--t0={t0}",
        f"--duration={duration}",
        f"--output_image_file={output_file}"
    ]

    print(f"🚀 Ejecutando modelo Astronet...")
    subprocess.run(cmd, check=True)
    print(f"✅ Predicción completada. Imagen guardada en {output_file}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    if len(sys.argv) != 2:
        print("Uso: python kepler_predict.py <KEPLER_ID>")
        sys.exit(1)

    kepler_id = sys.argv[1].strip()
    try:
        period, t0, duration = get_kepler_params(kepler_id)
        download_fits_files(kepler_id)
        run_astronet(kepler_id, period, t0, duration)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
