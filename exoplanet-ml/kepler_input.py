import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
from lightkurve import search_lightcurvefile, search_lightcurve

# ----------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/exoplanet-ml/MODEL_DIR")
KEPLER_DATA_DIR = os.environ.get("KEPLER_DATA_DIR", "/workspace/exoplanet-ml/KEPLER_DATA_DIR")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/exoplanet-ml/kepler_pictures")

os.makedirs(KEPLER_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# FUNCIONES PRINCIPALES
# ----------------------------
def get_lightcurve(kepler_id):
    """Descarga o carga la curva de luz de un Kepler ID."""
    print(f"🔍 Buscando lightcurve para KIC {kepler_id}...")

    try:
        lc_file = search_lightcurvefile(f"KIC {kepler_id}", mission="Kepler").download()
        lc = lc_file.PDCSAP_FLUX.remove_nans().normalize()
        print(f"✅ Curva de luz descargada con {len(lc.flux)} puntos.")
        return lc
    except Exception as e:
        print(f"❌ Error al obtener la curva de luz: {e}")
        return None


def find_transit_parameters(lc):
    """Usa BoxLeastSquares para encontrar parámetros de tránsito."""
    time = lc.time.value
    flux = lc.flux.value

    print("🔎 Buscando mejor período mediante BoxLeastSquares...")
    model = BoxLeastSquares(time, flux)
    periods = np.linspace(0.5, 30, 5000)
    results = model.power(periods, 0.1)

    best_period = results.period[np.argmax(results.power)]
    best_t0 = results.transit_time[np.argmax(results.power)]
    duration = results.duration[np.argmax(results.power)]

    print(f"🔭 Parámetros detectados:")
    print(f"   Periodo  = {best_period:.5f} días")
    print(f"   T0       = {best_t0:.5f}")
    print(f"   Duración = {duration * 24:.2f} horas")

    return best_period, best_t0, duration


def plot_lightcurve(lc, kepler_id, output_dir=OUTPUT_DIR):
    """Grafica y guarda la curva de luz."""
    plt.figure(figsize=(10, 4))
    plt.scatter(lc.time.value, lc.flux.value, s=1, color="black")
    plt.title(f"Kepler Lightcurve: {kepler_id}")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Flujo normalizado")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{kepler_id}_lightcurve.png")
    plt.savefig(output_path)
    plt.close()
    print(f"💾 Curva guardada en {output_path}")


def predict(kepler_id):
    """Pipeline completo para obtener, analizar y graficar una curva de luz."""
    lc = get_lightcurve(kepler_id)
    if lc is None:
        print("⚠️ No se pudo procesar la curva de luz.")
        return None

    period, t0, duration = find_transit_parameters(lc)
    plot_lightcurve(lc, kepler_id)
    print("✅ Proceso completado.")
    return period, t0, duration


# ----------------------------
# MODO DIRECTO (CLI)
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        kepler_id = sys.argv[1]
        predict(kepler_id)
    else:
        print("Uso: python kepler_input.py <KEPLER_ID>")
