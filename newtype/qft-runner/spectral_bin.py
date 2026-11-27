"""Selected-bin Fourier arithmetic for the explicit spectral watermark.

Uses NumPy only. Exact in real arithmetic, with floating-point roundoff.
This changes neither the signature nor its thresholds, and adds no security.
"""
from __future__ import annotations
import numpy as np


def tone(length, frequency, sign=1):
    """Unit complex DFT tone, reducing phase modulo one cycle."""
    if not isinstance(frequency, (int, np.integer)) or not 0 <= frequency < length:
        raise ValueError('frequency outside transform')
    return np.exp(sign*2j*np.pi*((np.arange(length,dtype=np.int64)*frequency)%length)/length)


def inject_conjugate_pair(latent, frequency, signature):
    """Real part of IFFT(FFT(latent) + two conjugate bin updates)."""
    x=np.asarray(latent); s=np.asarray(signature)
    if x.ndim!=2 or s.shape!=(x.shape[1],) or x.shape[0]==0:
        raise ValueError('shape mismatch')
    t=tone(x.shape[0],frequency)
    # DC and Nyquist: the two updates hit the same bin; adding both is correct.
    return x.real + (2/x.shape[0])*(t[:,None]*s[None,:]).real


def selected_bin(latent, frequency):
    """One unnormalized forward DFT row, without constructing other bins."""
    x=np.asarray(latent)
    if x.ndim!=2 or x.shape[0]==0:
        raise ValueError('expected nonempty matrix')
    return np.einsum('t,tk->k',tone(x.shape[0],frequency,sign=-1),x)
