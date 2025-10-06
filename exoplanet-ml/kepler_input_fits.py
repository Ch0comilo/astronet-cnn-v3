import os
import sys
import subprocess
import requests

# ----------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "MODEL_DIR"))
KEPLER_DATA_DIR = os.environ.get("KEPLER_DATA_DIR", os.path.join(BASE_DIR, "KEPLER_DATA_DIR"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "kepler_pictures"))
API_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# ----------------------------
# FUNCIONES
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

    return period, t0, duration / 24  # Convertir horas a días

def run_astronet_with_fits(fits_file: str, period: float, t0: float, duration: float):
    """
    Ejecuta el modelo Astronet directamente con un archivo FITS local.
    """
    kepler_id = os.path.splitext(os.path.basename(fits_file))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{kepler_id}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-m", "astronet.predict",
        "--model=AstroCNNModel",
        "--config_name=local_global",
        f"--model_dir={MODEL_DIR}",
        f"--kepler_data_dir={KEPLER_DATA_DIR}",
        f"--kepler_id={kepler_id}",
        f"--period={period}",
        f"--t0={t0}",
        f"--duration={duration}",
        f"--fits_file={fits_file}",
        f"--output_image_file={output_file}"
    ]

    print(f"🚀 Ejecutando Astronet para {fits_file} ...")
    subprocess.run(cmd, check=True)
    print(f"✅ Predicción completada. Imagen guardada en {output_file}")
    return output_file

def predict_fits_with_kepler_id(fits_file: str, kepler_id: str):
    """
    Pipeline completo: obtiene parámetros automáticamente y ejecuta Astronet
    con un FITS local.
    """
    period, t0, duration = get_kepler_params(kepler_id)
    return run_astronet_with_fits(fits_file, period, t0, duration)

# ----------------------------
# EJEMPLO DE USO
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python predict_fits_kepler.py <FITS_FILE> <KEPLER_ID>")
        sys.exit(1)

    fits_file = sys.argv[1]
    kepler_id = sys.argv[2]

    predict_fits_with_kepler_id(fits_file, kepler_id)
