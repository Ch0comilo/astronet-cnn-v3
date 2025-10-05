import sys
from kepler_input.core import predict

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python -m kepler_input <KEPLER_ID>")
        sys.exit(1)

    kepler_id = sys.argv[1]
    predict(kepler_id)
