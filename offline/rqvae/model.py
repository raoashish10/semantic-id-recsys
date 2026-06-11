"""Residual Quantized VAE (RQ-VAE) for semantic ID generation.

Each item embedding is encoded into `num_levels` discrete codebook indices
(semantic ID tokens).  Items with similar content share common prefix tokens,
forming a tree structure where deeper levels capture finer-grained attributes.

Architecture
------------
  encoder:  MLP  input_dim → hidden_dim
  quantizer: L residual VQ layers, each with K codebook entries of size hidden_dim
  decoder:  MLP  hidden_dim → input_dim  (reconstruction target)

Training objective
------------------
  L = recon_loss + commitment_loss

  recon_loss:      MSE between decoder output and original embedding
  commitment_loss: pulls encoder output toward nearest codebook vector and vice versa,
                   using the straight-through estimator so gradients flow to the encoder

Reference: "Recommender Systems with Generative Retrieval" (Rajput et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Single-level vector quantizer with straight-through gradient estimator."""

    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1 / num_codes, 1 / num_codes)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args
            z: (B, D) encoder output or residual from previous level

        Returns
            z_q_st:  (B, D) quantized vector with straight-through gradient
            z_q:     (B, D) actual nearest codebook vector, detached (used for residuals)
            indices: (B,)   codebook entry indices
            loss:    scalar commitment loss for this level
        """
        # Squared L2 distance: ||z - e||^2 = ||z||^2 - 2*z·e^T + ||e||^2
        # Shape: (B, K)
        dists = (
            z.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(dim=-1)
        )
        indices = dists.argmin(dim=-1)           # (B,)
        z_q = self.codebook(indices)             # (B, D) — no gradient through here

        # Commitment loss:
        #   encoder_loss  = ||z - sg(z_q)||^2   encoder commits to the codebook
        #   codebook_loss = ||sg(z) - z_q||^2   codebook moves toward encoder output
        loss = (
            self.commitment_cost * F.mse_loss(z_q.detach(), z)
            + F.mse_loss(z_q, z.detach())
        )

        # Straight-through estimator: copy gradient of z_q_st to z
        z_q_st = z + (z_q - z).detach()

        return z_q_st, z_q.detach(), indices, loss


class RQVAE(nn.Module):
    """
    Residual Quantized VAE.

    Encodes each item embedding into a tuple of L discrete tokens.
    The tuple is the item's semantic ID.

    The residual structure ensures each level captures what the previous
    levels failed to represent, so prefix tokens capture coarse-grained
    similarity and later tokens add fine-grained detail.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_codes: int = 256,
        num_levels: int = 3,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_codes, hidden_dim, commitment_cost)
            for _ in range(num_levels)
        ])

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Encode embeddings to semantic IDs (inference + training).

        Returns
            quantized_sum: (B, D)  sum of quantized vectors for decoder input
            codes:         list of L tensors each (B,) — the semantic ID tokens
            commit_loss:   scalar  total commitment loss across all levels
        """
        z = self.encoder(x)

        residual = z
        codes: list[torch.Tensor] = []
        quantized_st_list: list[torch.Tensor] = []
        commit_loss = torch.tensor(0.0, device=x.device)

        for quantizer in self.quantizers:
            z_q_st, z_q, idx, level_loss = quantizer(residual)
            codes.append(idx)
            quantized_st_list.append(z_q_st)
            # Next level quantizes the residual after actual (not straight-through) quantization
            residual = residual - z_q
            commit_loss = commit_loss + level_loss

        quantized_sum = torch.stack(quantized_st_list, dim=0).sum(dim=0)  # (B, D)
        return quantized_sum, codes, commit_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Training forward pass.

        Returns
            x_recon:    (B, input_dim)  reconstructed embedding
            codes:      list of L tensors each (B,)
            total_loss: scalar  recon_loss + commit_loss
        """
        quantized, codes, commit_loss = self.encode(x)
        x_recon = self.decoder(quantized)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, codes, recon_loss + commit_loss

    @torch.no_grad()
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Inference only: return semantic IDs as a (B, num_levels) int tensor."""
        _, codes, _ = self.encode(x)
        return torch.stack(codes, dim=-1)  # (B, L)

    def codebook_lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct quantized vector from codes without going through the encoder.

        Args
            codes: (B, L) integer tensor of semantic ID tokens

        Returns
            (B, hidden_dim) sum of codebook vectors
        """
        total = torch.zeros(codes.shape[0], self.hidden_dim, device=codes.device)
        for level, quantizer in enumerate(self.quantizers):
            total += quantizer.codebook(codes[:, level])
        return total
