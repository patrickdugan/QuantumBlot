# 🔮 QCTX – Quantum Context Toolkit

QCTX is an experimental toolkit for building **spectral fingerprints of text** using PCA + Quantum Fourier Transform (QFT)-style probes.  
It’s designed for analyzing large text corpora, extracting motifs, and experimenting with self-model accretion and narrative alignment.

---

## ✨ Features
- **Ingest & Preprocess**: clean and slice text/markdown into analyzable chunks.
- **Embeddings**: encode text with transformer models (via `sentence-transformers`).
- **Dimensionality Reduction**: project embeddings into smaller latent space with PCA.
- **QFT Fingerprints**: generate Fourier-style spectra over latent vectors.
- **Motif Analysis**: extract recurring narrative/semantic motifs.
- **CLI**: run all stages with a simple `typer` command-line interface.

---

## 📂 Project Structure
```
qctx/
  loader.py        # text ingest & preprocessing
  embeddings.py    # embedding generator
  dimred.py        # PCA reduction
  qft_pca.py       # QFT-style spectral probes
  motifs.py        # motif extraction
  cli_main.py      # Typer CLI entrypoint
data/
  sample_texts.txt # example input
```

---

## Optional CPU Fourier-shell screening (QB-COMPACT-01)

`qctx.shell_spectrum` computes additive-order Fourier energy without an FFT, with
no symmetry assumption on the input vector. The new shell descriptor is distinct
from existing contiguous FFT bands. It can aggregate measured counts with an
explicit bit order, or provide an opt-in upper-bound screen before cosine
reranking. The dense synthetic control shows no screening benefit, so the
existing retrieval default is unchanged.

```bash
python -m qctx.shell_spectrum embeddings.npy
python -m qctx.shell_spectrum embeddings.npy --query query.npy --k 20
```

Vectors must be finite, nonzero and explicitly padded to a power of two. The
screen retains the original normalized vectors for exact-score reranking;
`--no-screen` scans all rows. See [benchmark and proofs](experiments/qb_compact01/README.md).

## 🚀 Installation
```bash
git clone https://github.com/YOURNAME/qft-probes
cd qft-probes
pip install -r requirements.txt
```

Requirements:
- `typer[all]`
- `qiskit`
- `scikit-learn`
- optional: `sentence-transformers`

---

## 🔧 Usage Flow

### 1. Ingest raw text
```bash
python -m qctx.cli_main ingest ./data/sample_texts.txt --limit 2048
```

### 2. Encode to embeddings
```bash
python -m qctx.cli_main embed ./data/sample_texts.txt
```

### 3. Reduce with PCA
```bash
python -m qctx.cli_main pca ./embeddings.npy
```

### 4. Probe with QFT
```bash
python -m qctx.cli_main qft "[0,1,0,0]" --shots 512
```

### 5. Extract motifs
```bash
python -m qctx.cli_main motifs ./reduced.npy
```

---

## 📊 Example Output
- PCA projection plots (2D/3D)  
- QFT spectra (magnitude vs frequency bins)  
- Motif lists showing dominant narrative/semantic clusters  

---

## 🌍 Applications
- Context compression & motif analysis  
- Self-model stability under role shifts  
- Narrative benchmarking for multi-agent storyworlds  
- Exploratory tools for alignment research  

---

## 🛠 Roadmap
- [ ] Add Gradio demo (upload text → see fingerprint plot)
- [ ] HuggingFace dataset integration
- [ ] Multi-agent simulation probes

---

## 📑 License
MIT – free for research & collaboration
