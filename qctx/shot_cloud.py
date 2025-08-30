import os
import numpy as np
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService
from fourier_fingerprint import fourier_fingerprint_runtime  # make sure this is in your PYTHONPATH

# Load IBM Cloud credentials from qblot.env
load_dotenv("qblot.env")

service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=os.environ.get("IBM_CLOUD_API_KEY"),
    instance=os.environ.get("IBM_QUANTUM_CRN"),
)

# --- Load reduced PCA vector ---
vec = np.load("vectors_pca_topk.npy")   # <-- replace with your PCA output
n = len(vec)

# Ensure power-of-two length (pad with zeros if needed)
pow2 = 1 << (n - 1).bit_length()
if n < pow2:
    vec = np.pad(vec, (0, pow2 - n))

# Normalize amplitudes
amplitudes = vec / np.linalg.norm(vec)

# --- Run on IBM backend ---
counts = fourier_fingerprint_runtime(
    amplitudes,
    shots=2048,
    backend_name="ibm_brisbane",   # or any available backend in your account
)

print(counts)
