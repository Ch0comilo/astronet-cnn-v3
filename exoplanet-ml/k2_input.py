import os
import sys
import subprocess
import numpy as np
from lightkurve import search_lightcurve
from astropy.io import fits

# ----------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------
MODEL_DIR = "/workspace/exoplanet-ml/MODEL_DIR"
KEPLER_DATA_DIR = "/workspace/exoplanet-ml/KEPLER_DATA_DIR"
OUTPUT_DIR = "/workspace/exoplanet-ml/kepler_pictures"

# ----------------------------
# OBTENER Y CREAR ARCHIVO FITS
# ----------------------------
def create_k2_fits(k2_id: str):
    """Descarga lightcurve K2 y genera archivo FITS compatible con Astronet."""
    prefix = k2_id[:4]
    save_dir = os.path.join(KEPLER_DATA_DIR, prefix, k2_id)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔍 Buscando lightcurve para EPIC {k2_id}...")
    search_result = search_lightcurve(f"EPIC {k2_id}", mission='K2')

    if len(search_result) == 0:
        raise ValueError(f"No se encontraron datos para EPIC {k2_id}")

    lc = search_result[0].download()

    # Extraer datos
    data = lc.hdu[1].data
    time = data["TIME"]
    if "SAP_FLUX" in data.names:
        flux = data["SAP_FLUX"]
    elif "FLUX" in data.names:
        flux = data["FLUX"]
    else:
        raise KeyError("No se encontró ninguna columna de flujo compatible.")

    # Filtrar NaN e Inf
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]

    # Crear columnas Kepler-style
    cols = fits.ColDefs([
        fits.Column(name='TIME', format='E', array=time),
        fits.Column(name='PDCSAP_FLUX', format='E', array=flux)
    ])

    hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")

    # Guardar archivo
    output_file = os.path.join(save_dir, f"kplr{int(k2_id):09d}-2009259160929_llc.fits")
    hdu.writeto(output_file, overwrite=True)

    print(f"✅ Archivo FITS listo para Astronet: {output_file}")
    return output_file


# ----------------------------
# PARÁMETROS FICTICIOS
# ----------------------------
def generate_fake_params(k2_id: str):
    """Genera parámetros ficticios para EPIC (sin API oficial disponible)."""
    np.random.seed(int(k2_id) % 1000)
    period = np.random.uniform(1.5, 20.0)        # días
    t0 = np.random.uniform(2000, 2300)           # BKJD aproximado
    duration = np.random.uniform(1.0, 12.0) / 24 # días

    print(f"📊 Parámetros ficticios para EPIC {k2_id}:")
    print(f"   Period  = {period:.4f} días")
    print(f"   T0      = {t0:.4f} BKJD")
    print(f"   Duration= {duration*24:.4f} horas")

    return period, t0, duration


# ----------------------------
# EJECUTAR EL MODELO ASTRONET
# ----------------------------
def run_astronet(k2_id: str, period: float, t0: float, duration: float):
    """Ejecuta el modelo Astronet para el K2 ID."""
    output_file = os.path.join(OUTPUT_DIR, f"{k2_id}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        "python", "-m", "astronet.predict",
        "--model=AstroCNNModel",
        "--config_name=local_global",
        f"--model_dir={MODEL_DIR}",
        f"--kepler_data_dir={KEPLER_DATA_DIR}",
        f"--kepler_id={k2_id}",
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
        print("Uso: python k2_predict.py <K2_ID>")
        sys.exit(1)

    k2_id = sys.argv[1].strip()
    try:
        period, t0, duration = generate_fake_params(k2_id)
        create_k2_fits(k2_id)
        run_astronet(k2_id, period, t0, duration)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
