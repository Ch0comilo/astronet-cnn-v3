import os
import sys
import subprocess
import numpy as np
from lightkurve import search_lightcurve, search_lightcurvefile
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------
MODEL_DIR = "/workspace/exoplanet-ml/MODEL_DIR"
TESS_DATA_DIR = "/workspace/exoplanet-ml/KEPLER_DATA_DIR"  # misma ruta que Kepler
OUTPUT_DIR = "/workspace/exoplanet-ml/kepler_pictures"

# ----------------------------
# DESCARGA LIGHTCURVE Y CÁLCULO DE PARÁMETROS
# ----------------------------
def analyze_tess_lightcurve(tic_id: str):
    """Intenta obtener periodo, t0 y duración usando BLS."""
    print(f"🔍 Buscando lightcurve para TIC {tic_id}...")
    results = search_lightcurvefile(f"TIC {tic_id}", mission="TESS")

    if len(results) == 0:
        raise ValueError("No se encontraron observaciones para este TIC.")

    lc = results[0].download().PDCSAP_FLUX.remove_nans().normalize()
    time = lc.time.value
    flux = lc.flux.value

    if len(time) < 100:
        raise ValueError("Demasiados pocos datos válidos para análisis BLS.")

    print("⚙️  Ejecutando análisis Box Least Squares...")
    periods = np.linspace(0.5, 30, 20000)
    bls = BoxLeastSquares(time, flux)
    bls_result = bls.power(periods, 0.1)

    best = np.argmax(bls_result.power)
    period = bls_result.period[best]
    t0 = bls_result.transit_time[best]
    duration = bls_result.duration[best]

    print(f"✅ Parámetros estimados para TIC {tic_id}:")
    print(f"   Periodo : {period:.5f} días")
    print(f"   T0      : {t0:.5f}")
    print(f"   Duración: {duration*24:.2f} horas")

    # Guardar gráfico BLS
    plt.figure(figsize=(7, 3))
    plt.plot(bls_result.period, bls_result.power, lw=0.7)
    plt.xlabel("Periodo (días)")
    plt.ylabel("Potencia BLS")
    plt.title(f"BLS TIC {tic_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tic_id}_bls_power.png"))
    plt.close()

    return float(period), float(t0), float(duration)/24

# ----------------------------
# CREAR ARCHIVO FITS
# ----------------------------
def create_tess_fits(tic_id: str):
    """Descarga lightcurve TESS y genera archivo FITS compatible con Astronet."""
    # Normalizar el ID a 9 dígitos (rellenar con ceros a la izquierda)
    tic_id_padded = f"{int(tic_id):09d}"

    prefix = tic_id_padded[:4]
    save_dir = os.path.join(TESS_DATA_DIR, prefix, tic_id_padded)
    os.makedirs(save_dir, exist_ok=True)

    print(f"📦 Generando archivo FITS para TIC {tic_id_padded}...")
    search_result = search_lightcurve(f"TIC {tic_id}", mission='TESS')
    if len(search_result) == 0:
        raise ValueError(f"No se encontraron datos para TIC {tic_id}")

    lc = search_result[0].download()
    data = lc.hdu[1].data
    time = data["TIME"]
    if "SAP_FLUX" in data.names:
        flux = data["SAP_FLUX"]
    elif "FLUX" in data.names:
        flux = data["FLUX"]
    else:
        raise KeyError("No se encontró ninguna columna de flujo compatible.")

    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]

    cols = fits.ColDefs([
        fits.Column(name='TIME', format='E', array=time),
        fits.Column(name='PDCSAP_FLUX', format='E', array=flux)
    ])
    hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")

    output_file = os.path.join(save_dir, f"kplr{tic_id_padded}-2009259160929_llc.fits")
    hdu.writeto(output_file, overwrite=True)

    print(f"✅ Archivo FITS listo: {output_file}")
    return output_file, tic_id_padded

# ----------------------------
# PARÁMETROS FICTICIOS (fallback)
# ----------------------------
def generate_fake_params(tic_id: str):
    """Genera parámetros ficticios para TIC (si BLS no tiene señal clara)."""
    np.random.seed(int(tic_id) % 1000)
    period = np.random.uniform(1.5, 15.0)
    t0 = np.random.uniform(1300, 1500)
    duration = np.random.uniform(1.0, 10.0) / 24
    print(f"⚠️  Usando parámetros ficticios:")
    print(f"   Period  = {period:.4f} días")
    print(f"   T0      = {t0:.4f} BKJD")
    print(f"   Duration= {duration*24:.4f} horas")
    return period, t0, duration

# ----------------------------
# EJECUTAR MODELO ASTRONET
# ----------------------------
def run_astronet(tic_id_padded: str, period: float, t0: float, duration: float):
    output_file = os.path.join(OUTPUT_DIR, f"{tic_id_padded}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        "python", "-m", "astronet.predict",
        "--model=AstroCNNModel",
        "--config_name=local_global",
        f"--model_dir={MODEL_DIR}",
        f"--kepler_data_dir={TESS_DATA_DIR}",
        f"--kepler_id={tic_id_padded}",
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
        print("Uso: python tess_predict.py <TIC_ID>")
        sys.exit(1)

    tic_id = sys.argv[1].strip()
    try:
        try:
            period, t0, duration = analyze_tess_lightcurve(tic_id)
        except Exception as e:
            print(f"⚠️  No se pudieron estimar parámetros reales: {e}")
            period, t0, duration = generate_fake_params(tic_id)

        fits_path, tic_id_padded = create_tess_fits(tic_id)
        run_astronet(tic_id_padded, period, t0, duration)

    except Exception as e:
        print(f"❌ Error general: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
