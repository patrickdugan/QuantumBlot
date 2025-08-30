from qiskit_aer import AerSimulator
from qiskit import transpile
backend = AerSimulator()
job = backend.run(transpile(build_qft_measured(amplitudes), backend), shots=2048)
counts = job.result().get_counts()