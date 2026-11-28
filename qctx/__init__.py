"""QCTX exports, loaded on demand so NumPy-only probes need no IBM SDK."""
from importlib import import_module

_EXPORTS = {
    'slurp_text': 'loader', 'preprocess_markdown': 'loader', 'slice_bytes': 'loader',
    'fourier_fingerprint': 'qft_pca', 'pca_project': 'dimred',
    'encode_texts': 'embeddings', 'pull_motifs': 'motifs',
    'shell_probabilities': 'shell_spectrum', 'counts_shell_probabilities': 'shell_spectrum',
    'shell_coordinates': 'shell_spectrum', 'shell_transform': 'shell_spectrum',
    'ShellIndex': 'shell_spectrum',
    'LearnedScreen': 'learned_screen', 'fit_basis': 'learned_screen',
    'ProgressiveScreen': 'progressive_screen',
    'NativeProgressiveScreen': 'native_screen',
    'RankPolicy': 'rank_policy', 'fit_policy': 'rank_policy',
    'BatchScreen': 'batch_screen',
    'QuantizedBatchScreen': 'quantized_screen',
    'NativeInt8Screen': 'quantized_screen',
    'DiskScreen': 'disk_screen', 'build_store': 'disk_screen',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f'.{_EXPORTS[name]}', __name__), name)
    globals()[name] = value
    return value
