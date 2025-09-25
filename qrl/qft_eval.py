# qft_eval.py
import torch
from qft_trace_regularizer import QFTConfig, MotifBank, QFTTraceRegularizer

def make_default(cfg: QFTConfig, bands=16, motifs=None, learnable=False):
    if motifs is None:
        # three handy priors: low-freq dominant, mid-band iterative, spike-then-stop
        m = torch.zeros(3, bands)
        m[0, 0] = 1.0
        m[1, bands//3:2*bands//3] = 1.0
        m[2, :3] = torch.tensor([0.8, 0.4, 0.2])
        motifs = m
    bank = MotifBank(num_motifs=motifs.size(0), bands=motifs.size(1), learnable=learnable, init=motifs)
    qft = QFTTraceRegularizer(cfg, bank)
    return qft

@torch.no_grad()
def quick_eval():
    cfg = QFTConfig(bands=16)
    qft = make_default(cfg)
    B, T, D = 8, 128, 64
    # fake hidden states with different “structures”
    H = torch.randn(B, T, D)
    # add a low-frequency sweep on half the batch to mimic “clean reasoning”
    t = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1)
    H[:B//2] += torch.sin(2 * torch.pi * 2 * t)  # 2 cycles over T
    out = qft(H)
    print("aux_loss:", float(out["aux_loss"]))
    print("reward_bonus mean:", float(out["reward_bonus"].mean()))
    print("router_feats shape:", tuple(out["router_feats"].shape))

if __name__ == "__main__":
    quick_eval()
