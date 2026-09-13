from setuptools import setup, find_packages

setup(
    name="true_tone",
    version="1.0.0",
    description="Real-Time SIM-to-SIM Call Deepfake Voice Detection Framework",
    author="True Tone Team (SIH)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "websockets>=12.0",
        "python-multipart>=0.0.9",
        "pytest>=7.4.0",
    ],
)
