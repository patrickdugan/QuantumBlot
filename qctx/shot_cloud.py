# Put these in your env or pass as args:
# export IBM_CLOUD_API_KEY="...your key..."
# export IBM_QUANTUM_CRN="crn:v1:bluemix:public:quantum-computing:...:..."

from dotenv import load_dotenv
load_dotenv("qblot.env")

service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=token or os.environ.get("IBM_CLOUD_API_KEY"),
    instance=instance or os.environ.get("IBM_QUANTUM_CRN"),
)

amplitudes = np.ones(256)  # example; your centroid or PC1 amplitudes go here (already normalized to power-of-two length)
counts = fourier_fingerprint_runtime(
    amplitudes,
    shots=2048,
    backend_name="ibm_brisbane",
)
print(counts)
