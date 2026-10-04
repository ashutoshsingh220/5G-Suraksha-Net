"""Temporal fight classifier model — the ONE definition shared by training and inference.

WHY this module exists: the GRU used to be defined *inside*
`TorchTemporalClassifier.__init__`, so training code could not import the same
architecture. A checkpoint trained against a copy-pasted twin would load only by
accident, and any drift between the two would be invisible until inference. Both
sides now build the network from `build_gru()`.

Architecture is UNCHANGED from the original inference definition — hyperparameters
were read off the existing code, not chosen here:

    GRU(input_size=8, hidden_size=64, num_layers=1, batch_first=True,
        bidirectional=False, dropout=0)
      -> pool = mean over time + last time step        (64 + 64 -> 64)
      -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> squeeze
      -> RAW LOGIT, shape (B,)

`torch.sigmoid` is applied by the caller, never inside `forward`. That is what
makes BCEWithLogitsLoss the correct training loss.
"""
from __future__ import annotations

FEATURE_DIM = 8          # FightCandidateDetector.FEATURE_DIM — per-frame pair kinematics
HIDDEN_SIZE = 64
HEAD_HIDDEN = 32
NUM_LAYERS = 1
DROPOUT = 0.0
BIDIRECTIONAL = False


def build_gru(feature_dim: int = FEATURE_DIM, hidden_size: int = HIDDEN_SIZE):
    """Construct the temporal GRU. torch is imported lazily (module import stays cheap)."""
    import torch
    from torch import nn

    class TemporalGRU(nn.Module):
        """(B, T, feature_dim) -> (B,) raw fight logit."""

        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.feature_dim = dim
            self.hidden_size = hidden
            self.num_layers = NUM_LAYERS
            self.bidirectional = BIDIRECTIONAL
            self.dropout = DROPOUT
            self.gru = nn.GRU(dim, hidden, batch_first=True, num_layers=NUM_LAYERS)
            self.head = nn.Sequential(
                nn.Linear(hidden, HEAD_HIDDEN), nn.ReLU(), nn.Linear(HEAD_HIDDEN, 1)
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if x.dim() != 3 or x.size(-1) != self.feature_dim:
                raise ValueError(
                    f"expected (BATCH, T, {self.feature_dim}) features, got {tuple(x.shape)}"
                )
            out, _ = self.gru(x)
            pooled = out.mean(dim=1) + out[:, -1, :]   # mean + last
            return self.head(pooled).squeeze(-1)

        def architecture(self) -> dict:
            return {
                "type": "gru_mean_last_pool_mlp",
                "input_size": self.feature_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "head": [self.hidden_size, HEAD_HIDDEN, 1],
                "output": "raw logit (apply sigmoid for probability)",
            }

    return TemporalGRU(feature_dim, hidden_size)
