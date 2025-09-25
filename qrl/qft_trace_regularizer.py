# qft_trace_regularizer.py
# MIT/Apache-2.0 style snippet — use freely.

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _spectral_entropy(mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """mag: [..., K] nonnegative magnitudes"""
    p = mag / (mag.sum(dim=-1, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(dim=-1)

def _hann_window(T: int, device: torch.device) -> torch.Tensor:
    n = torch.arange(T, device=device)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * n / (T - 1))

@dataclass
class QFTConfig:
    bands: int = 16                 # number of rFFT bins to keep (low-frequency focus)
    pool: str = "mean"              # "mean" | "cls" | "attn_diag" (see notes)
    aux_loss_weight: float = 0.05   # λ for auxiliary spectral loss
    entropy_weight: float = 0.1     # λ2 inside aux loss (penalize spectral entropy)
    reward_weight: float = 0.2      # w scaling r_qft added to task reward
    entropy_penalty: float = 0.2    # w2 inside r_qft (subtract entropy)
    normalize_mag: bool = True      # L1 normalize spectrum before similarity
    detach_reward: bool = True      # keep reward term out of grad (typical for RL)
    use_window: bool = True         # apply Hann window before rFFT
    eps: float = 1e-8

class MotifBank(nn.Module):
    """
    Holds a small set of spectral 'motifs' (K bands each).
    You can init with fixed motifs or let them be learnable.
    """
    def __init__(self, num_motifs: int, bands: int, learnable: bool = False, init: Optional[torch.Tensor] = None):
        super().__init__()
        if init is None:
            init = torch.randn(num_motifs, bands) * 0.01
            init[:, 0] = 1.0  # bias to low-freq dominance as a sane default
        assert init.shape == (num_motifs, bands)
        self.motifs = nn.Parameter(init, requires_grad=learnable)

    def forward(self) -> torch.Tensor:
        # return normalized motifs (L2) to stabilize cosine sims
        m = self.motifs
        return F.normalize(m, dim=-1)

class QFTTraceRegularizer(nn.Module):
    """
    Plug this into your trainer. Feed it a time-series per sample; it returns:
      - aux loss term (to add to SFT/critic loss)
      - reward bonus (to add in RL)
      - features for router (entropy, best motif score, etc.)
    """
    def __init__(self, cfg: QFTConfig, motif_bank: MotifBank):
        super().__init__()
        self.cfg = cfg
        self.motifs = motif_bank

    def _pool_series(self, H: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        H: [B, T, D] hidden states (or other scalar stream you log per token).
        A: optional attention maps [B, L, Hh, T, T] — if provided and cfg.pool == "attn_diag",
           we average layers/heads and take the main diagonal as a scalar time series.
        Returns x: [B, T] pooled scalar sequence.
        """
        pool = self.cfg.pool
        if pool == "mean":
            # mean over feature dim -> scalar per token
            return H.mean(dim=-1)
        elif pool == "cls":
            # take a single channel (first dim) as proxy; or H[...,0]
            return H[..., 0]
        elif pool == "attn_diag":
            assert A is not None, "Attention tensor required for attn_diag pooling."
            # average layers and heads; take diag energy per time
            B, L, Hh, T, _ = A.shape
            attn = A.mean(dim=(1,2))  # [B, T, T]
            diag = attn.diagonal(dim1=-2, dim2=-1)  # [B, T]
            return diag
        else:
            raise ValueError(f"Unknown pool: {pool}")

    def _qft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T] scalar time series
        Returns (mag_K, entropy): [B, K], [B]
        """
        B, T = x.shape
        if self.cfg.use_window:
            w = _hann_window(T, x.device)
            x = x * w
        # rFFT over time, keep low-K bands
        Ffull = torch.fft.rfft(x, dim=1)  # [B, T//2+1]
        mag = Ffull.abs()
        K = min(self.cfg.bands, mag.shape[1])
        mag = mag[:, :K]
        if self.cfg.normalize_mag:
            mag = mag / (mag.sum(dim=-1, keepdim=True) + self.cfg.eps)
        ent = _spectral_entropy(mag, eps=self.cfg.eps)  # [B]
        return mag, ent

    def _motif_scores(self, mag: torch.Tensor) -> torch.Tensor:
        """
        mag: [B, K] (optionally L1 normalized)
        returns cosine similarity to each motif: [B, M]
        """
        motifs = self.motifs()  # [M, K], L2-normalized
        mag_n = F.normalize(mag, dim=-1)
        return mag_n @ motifs.t()

    def forward(
        self,
        H: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        # alpha: optional coefficients if you want to project onto motif subspace; otherwise we use max motif score
    ) -> Dict[str, torch.Tensor]:
        """
        Inputs:
          H: [B, T, D] hidden states (or any scalarizable per-token signal)
          A: optional attention tensor for pooling mode "attn_diag"
        Outputs dict:
          {
            "aux_loss":       scalar,
            "reward_bonus":   [B] (detach by default),
            "router_feats":   [B, F]  (# F = 2 + num_motifs: [entropy, best_sim, sims...])
            "qft_mag":        [B, K],
            "qft_entropy":    [B],
            "motif_sims":     [B, M]
          }
        """
        cfg = self.cfg
        x = self._pool_series(H, A)              # [B, T]
        mag, entropy = self._qft(x)              # [B, K], [B]
        sims = self._motif_scores(mag)           # [B, M]
        best_sim, _ = sims.max(dim=-1)           # [B]

        # Aux loss: encourage spectra to match motif subspace + mild entropy penalty
        if alpha is not None:
            # Fit to motif subspace: ||mag - M @ alpha||^2
            motifs = self.motifs()               # [M, K]
            recon = alpha @ motifs               # [B, K]  (requires alpha shape [B, M])
            recon = F.normalize(recon, dim=-1)
            L_fit = F.mse_loss(F.normalize(mag, dim=-1), recon, reduction="mean")
        else:
            # Use (1 - best cosine sim) as a simple fit loss
            L_fit = (1.0 - best_sim).mean()

        L_ent = entropy.mean()
        aux_loss = cfg.aux_loss_weight * (L_fit + cfg.entropy_weight * L_ent)

        # Reward bonus (shape [B]): r_qft = best_sim - w * entropy
        r_qft = best_sim - cfg.entropy_penalty * entropy
        if cfg.detach_reward:
            r_qft = r_qft.detach()
        reward_bonus = cfg.reward_weight * r_qft

        # Router features: concat entropy, best_sim, full sims
        router_feats = torch.cat([entropy.unsqueeze(-1), best_sim.unsqueeze(-1), sims], dim=-1)

        return {
            "aux_loss": aux_loss,
            "reward_bonus": reward_bonus,
            "router_feats": router_feats,
            "qft_mag": mag,
            "qft_entropy": entropy,
            "motif_sims": sims,
        }
