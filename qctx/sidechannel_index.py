import numpy as np

def make_reserved_indices(D=512, R=16, seed=1337):
    # pick R tail dims; permute by key
    base = np.arange(D-R, D)
    rng = np.random.default_rng(seed)
    rng.shuffle(base)
    return base  # length R

def simhash_bits(z, K=128, seed=4242):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, z.shape[-1]))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    a = C @ z
    bits = (a >= 0).astype(np.uint8)
    return bits  # [K]

def embed_sidechannel(vec512, bits, reserved_idx, epsilon=1e-3, mode='byte'):
    v = vec512.copy()
    D = v.shape[0]
    mask = np.ones(D, dtype=bool)
    mask[reserved_idx] = False
    # renorm only the real dims
    nrm = np.linalg.norm(v[mask]) + 1e-12
    v[mask] /= nrm
    # write signature
    R = reserved_idx.shape[0]
    if mode == 'byte':
        # pack bits -> bytes, write as (b+0.5)/256 scaled around 0 with epsilon
        B = (bits.reshape(-1,8) * (1 << np.arange(8))).sum(axis=1).astype(np.uint8)
        B = B[:R] if B.shape[0] >= R else np.pad(B, (0, R - B.shape[0]), 'constant')
        payload = (B.astype(np.float32) + 0.5) / 256.0  # 0..~1
        payload = (payload - 0.5) * 2.0 * epsilon       # ~[-ε, +ε]
        v[reserved_idx] = payload
    else:
        # binary: one bit per dim (use first R bits)
        b = bits[:R]
        v[reserved_idx] = (b * 2 - 1) * epsilon
    return v

def extract_sidechannel(vec512, reserved_idx, mode='byte', epsilon=1e-3, K=128):
    x = vec512[reserved_idx]
    if mode == 'byte':
        # invert the mapping
        payload = (x / (2*epsilon)) + 0.5
        payload = np.clip(payload, 0.0, 1.0)
        B = np.round(payload * 256 - 0.5).astype(np.uint8)
        # bytes -> bits
        bits = np.unpackbits(B).astype(np.uint8)
        return bits[:K]
    else:
        return (vec512[reserved_idx] >= 0).astype(np.uint8)

def cosine_masked(a, b, reserved_idx):
    mask = np.ones_like(a, dtype=bool)
    mask[reserved_idx] = False
    a2, b2 = a[mask], b[mask]
    return float((a2 @ b2) / ((np.linalg.norm(a2)+1e-12)*(np.linalg.norm(b2)+1e-12)))
