# Quantum Context Toolkit (scrambled reimplementation)

This is a clean-room, differently named/flow reimplementation of your earlier modules:

- `ingest.py` -> `qctx/loader.py` (`slurp_text`, `preprocess_markdown`, `slice_bytes`)
- `quantum_pca.py` -> `qctx/qft_pca.py` (`fourier_fingerprint`)
- `reduce_pca.py` -> `qctx/dimred.py` (`pca_project`)
- `embed.py` -> `qctx/embeddings.py` (`encode_texts`)
- `extract_theme.py` -> `qctx/motifs.py` (`pull_motifs`)
- `cli.py` -> `qctx/cli_main.py` (Typer CLI with subcommands)

Install Typer dependencies to run CLI: `pip install typer[all] qiskit scikit-learn` (and optionally `sentence-transformers`).

Example:
```
python -m qctx.cli_main ingest /path/to/file.txt --limit 2048
python -m qctx.cli_main qft "[0,1,0,0]" --shots 512
```
