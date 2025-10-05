FROM python:3.7-slim

# --- Dependencias del sistema ---
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Actualizar pip ---
RUN pip install --upgrade pip setuptools wheel

# --- Instalar TensorFlow 1.15 (CPU) ---
RUN pip install tensorflow==1.15

# --- Librerías comunes ---
RUN pip install numpy scipy pandas matplotlib scikit-learn astropy pydl

# --- Librerías adicionales útiles ---
RUN pip install jupyter notebook tensorboard

# --- Directorio de trabajo ---
WORKDIR /workspace

# --- Puerto para TensorBoard ---
EXPOSE 6006

# --- Comando por defecto: abrir bash ---
CMD ["/bin/bash"]
