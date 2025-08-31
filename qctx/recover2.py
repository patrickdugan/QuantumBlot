import os
import re
import json
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService

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
                os.environ[k] = v

load_env_file("qblot.env")
load_env_file(".env")

# ---------- connect service ----------
service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=os.environ.get("IBM_CLOUD_API_KEY"),
    instance=os.environ.get("IBM_QUANTUM_CRN"),
)

# ---------- recover ----------
job_id = os.environ.get("RECOVER_JOB_ID", "d2psqg494j0s73a458c0")
job = service.job(job_id)
print("Recovered job:", job.job_id(), "status:", job.status())

res = job.result()

# ---------- parse results ----------
for idx, pub in enumerate(res):
    counts = None
    try:
        # Preferred API
        if hasattr(pub.data, "meas") and hasattr(pub.data.meas, "get_counts"):
            counts = pub.data.meas.get_counts()
        else:
            # If multiple classical registers, combine them
            counts = pub.join_data().get_counts()
    except Exception as e:
        print(f"DEBUG: Failed standard get_counts → {e}")
        print("DEBUG: pub.data =", repr(pub.data))

    if counts is None:
        print("DEBUG: Could not extract counts.")
    else:
        out_path = f"job_{job.job_id()}_pub{idx}_counts.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2)
        print(f"Saved full counts to {out_path} (total {sum(counts.values())} shots)")
