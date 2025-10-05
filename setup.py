from setuptools import setup, find_packages

setup(
    name="kepler-input",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
    ],
    python_requires=">=3.8",
    author="Daniel",
    description="Módulo para descargar datos Kepler y ejecutar Astronet automáticamente",
    url="https://github.com/Ch0comilo/astronet-cnn-v3",
)
