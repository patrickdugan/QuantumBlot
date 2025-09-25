# pl_qft_callback.py
# MIT/Apache-2.0

from typing import Callable, Optional, Dict, Any
import torch
import pytorch_lightning as pl
from qft_trace_regularizer import QFTConfig, MotifBank, QFTTraceRegularizer

class QFTLightningCallback(pl.Callback):
    """
    Plug into Trainer(callbacks=[...]).
    Expects your LightningModule to call `self.qft_reward_bonus = ...` if you want RL shaping,
    but for SFT it will automatically add aux loss.

    Args:
      extract_hidden: fn(outputs, batch, batch_idx) -> Tensor [B,T,D]
                      how to get hidden states from your model's training_step outputs
      bands/motifs/etc via cfg or bank_init
    """
    def __init__(
        self,
        extract_hidden: Callable[[Dict[str,Any], Any, int], torch.Tensor],
        cfg: Optional[QFTConfig] = None,
        bank_init: Optional[torch.Tensor] = None,
        learnable_bank: bool = False,
        log_prefix: str = "qft/"
    ):
        super().__init__()
        self.cfg = cfg or QFTConfig()
        self.bank = MotifBank(
            num_motifs=(bank_init.size(0) if bank_init is not None else 3),
            bands=(bank_init.size(1) if bank_init is not None else self.cfg.bands),
            learnable=learnable_bank,
            init=bank_init
        )
        self.reg = QFTTraceRegularizer(self.cfg, self.bank)
        self.extract_hidden = extract_hidden
        self.log_prefix = log_prefix

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        `outputs` should include what your training_step returns.
        For SFT/critic training, ensure training_step stores base loss at `outputs["loss"]`
        or pl_module.training_step returns a dict with "loss".
        """
        if outputs is None:
            return
        # 1) Get hidden states
        H = self.extract_hidden(outputs, batch, batch_idx)  # [B,T,D]
        if H is None:
            return

        # 2) Compute QFT terms
        with torch.set_grad_enabled(True):
            q = self.reg(H=H)  # dict with aux_loss/reward_bonus/router_feats...

        # 3) Add aux loss (SFT/critic)
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"] + q["aux_loss"]
            outputs["loss"] = loss
        else:
            # If your step returns a tensor, add to pl_module.manual_loss or expose a hook.
            if hasattr(pl_module, "manual_loss"):
                pl_module.manual_loss = pl_module.manual_loss + q["aux_loss"]

        # 4) Optional: stash reward bonus for RL code (actor-critic/GRPO step)
        pl_module.qft_reward_bonus = q["reward_bonus"].detach()

        # 5) Log a few metrics
        ent = q["qft_entropy"].mean()
        feats = q["router_feats"].mean(dim=0)
        pl_module.log(self.log_prefix+"aux_loss", q["aux_loss"], prog_bar=False, on_step=True, on_epoch=True)
        pl_module.log(self.log_prefix+"entropy", ent, prog_bar=False, on_step=True, on_epoch=True)
        pl_module.log(self.log_prefix+"best_sim", feats[1], prog_bar=False, on_step=True, on_epoch=True)
