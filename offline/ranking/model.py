"""Point-wise MLP ranker.

Takes a (user_embedding, item_embedding) pair and predicts a relevance score.
Trained with BCE loss on positive (interacted) and sampled-negative item pairs.

The user embedding is the mean of the embeddings of their recent session items —
the same representation used to query the FAISS index. This means the ranker
learns to refine what the ANN already found, not to re-learn retrieval.
"""

import torch
import torch.nn as nn


class Ranker(nn.Module):
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """
        Args
            user_emb: (B, D)
            item_emb: (B, D)
        Returns
            logits: (B,) — pass through sigmoid for probability
        """
        return self.net(torch.cat([user_emb, item_emb], dim=-1)).squeeze(-1)
