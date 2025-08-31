# recover_job.py
import os
from qiskit_ibm_runtime import QiskitRuntimeService

from pathlib import Path
import re

# ---------- load your env ----------
def load_env_file(path: str):
    if not Path(path).exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
            if m:
                k, v = m.group(1), m.group(2)
                if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
                    v = v[1:-1]
                os.environ.setdefault(k, v)

load_env_file("qblot.env")
load_env_file(".env")

# ---------- connect service ----------
service = QiskitRuntimeService(
    channel="ibm_cloud",  # stick to ibm_cloud
    token=os.environ.get("IBM_CLOUD_API_KEY", "YOUR_API_KEY_HERE"),
    instance=os.environ.get("IBM_QUANTUM_CRN", "crn:v1:bluemix:public:quantum-computing:us-east:a/a237ff74999c4c69bf59f4d8a25f8786:db4b2706-3607-43b7-84bb-136b2ad34512::"),
)

# ---------- recover ----------
job_id = "d2ppuqs94j0s73a42gvg"
job = service.job(job_id)
print("Recovered job:", job.job_id(), "status:", job.status())

res = job.result()

# ---------- parse results ----------
for idx, pub in enumerate(res):
    data = getattr(pub, "data", pub)

    # Try quasi distribution
    quasi = None
    if hasattr(data, "get"):  # dict-like
        quasi = data.get("quasi_dist", None)

    counts = None
    if quasi is not None:
        shots = getattr(res, "execution_counts", 2048)
        counts = {k: int(round(v * shots)) for k, v in quasi.items()}
    else:
        # fallback: try get_counts or manual BitArray conversion
        if hasattr(data, "get_counts"):
            counts = data.get_counts()
        elif hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
            counts = data.meas.get_counts()
        elif hasattr(data, "meas"):
            # assume BitArray-like
            bitstrings = [str(b) for b in data.meas]
            counts = {}
            for b in bitstrings:
                counts[b] = counts.get(b, 0) + 1

import json

# after building counts
out_path = f"job_{job_id}_pub{idx}_counts.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(counts, f, indent=2)

print(f"Saved full counts to {out_path} (total {sum(counts.values())} shots)")
