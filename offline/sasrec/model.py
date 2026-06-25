"""SASRec — Self-Attentive Sequential Recommendation with Semantic IDs.

Items are no longer represented by opaque integer IDs.  Instead, each item
is represented by its L-token semantic ID from the RQ-VAE.  The embedding
for an item is the sum of its L codebook token embeddings, with level-
specific offsets so token 5 at level 0 is distinct from token 5 at level 1.

Architecture
------------
  item_embed:  sum of L token embeddings (each token offset by level * num_codes)
  positional:  learned positional embedding up to max_len
  transformer: N causal self-attention layers (left-to-right, no future leakage)
  output head: predict next item's semantic ID tokens (L classification heads)

Training loss: cross-entropy over num_codes classes at each of the L levels,
               averaged across levels and sequence positions.

Reference: "Self-Attentive Sequential Recommendation" (Kang & McAuley, 2018)
           extended with semantic IDs from Rajput et al. (2023)
"""

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(
        self,
        num_codes: int,
        num_levels: int,
        hidden_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # num_codes tokens per level; level l uses token range [l*num_codes, (l+1)*num_codes)
        vocab_size = num_codes * num_levels
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len + 1, hidden_dim)  # 0 reserved for padding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # One classification head per level — each predicts which of num_codes to use
        self.output_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_codes) for _ in range(num_levels)]
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def item_embedding(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, L) semantic IDs to (B, T, D) item embeddings.

        Each level's tokens are offset so they don't collide in the shared vocabulary.
        """
        # codes: (B, T, L)
        offsets = torch.arange(self.num_levels, device=codes.device) * self.num_codes
        adjusted = codes + offsets  # (B, T, L)
        embeds = self.token_embed(adjusted)  # (B, T, L, D)
        return embeds.sum(dim=-2)  # (B, T, D)

    def forward(
        self,
        codes: torch.Tensor,  # (B, T, L) — input sequence semantic IDs
        padding_mask: torch.Tensor,  # (B, T)    — True where position is padding
    ) -> list[torch.Tensor]:
        """
        Returns a list of L logit tensors, each (B, T, num_codes),
        representing the predicted semantic ID tokens for the *next* position.
        """
        B, T, _ = codes.shape

        # Item embeddings
        x = self.item_embedding(codes)  # (B, T, D)

        # Positional embeddings (1-indexed; 0 is for padding)
        positions = torch.arange(1, T + 1, device=codes.device).unsqueeze(0)  # (1, T)
        x = self.dropout(x + self.pos_embed(positions))

        # Causal attention mask: position i cannot attend to positions > i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=codes.device
        )

        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        h = torch.nan_to_num(h, nan=0.0)  # padding positions with all-masked keys produce nan
        # h: (B, T, D)

        return [head(h) for head in self.output_heads]  # L × (B, T, num_codes)

    @torch.no_grad()
    def recommend(
        self,
        codes: torch.Tensor,  # (B, T, L) session as semantic IDs
    ) -> torch.Tensor:
        """Predict the next item's semantic IDs. Returns (B, L) of predicted codes."""
        self.eval()
        padding_mask = torch.zeros(
            codes.shape[:2], dtype=torch.bool, device=codes.device
        )
        logits_per_level = self.forward(codes, padding_mask)  # L × (B, T, num_codes)
        # Take the prediction at the last position
        predicted = torch.stack(
            [logits[:, -1, :].argmax(dim=-1) for logits in logits_per_level],
            dim=-1,
        )  # (B, L)
        return predicted
