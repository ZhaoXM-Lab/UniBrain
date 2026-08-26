#!/usr/bin/env python3
"""
=============================================================================
File      : 00_vqvae_module.py
Directory : 10_uniBrain agent
Purpose   : Shared VQVAE components: NB loss, EMA vector quantizer, VQVAE model,
            and encoding/decoding helpers used by both training and inference.
=============================================================================
"""

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple


# ===========================================================================
# Negative Binomial NLL loss
# ===========================================================================
def nb_nll(x: Tensor, mu: Tensor, theta: Tensor, eps: float = 1e-8,
           reduction: str = "mean") -> Tensor:
    """Negative Binomial negative log-likelihood.

    Args:
        x: observed counts, shape [B, D], values >= 0
        mu: predicted mean, shape [B, D], values > 0
        theta: dispersion, shape [D] or [B, D], values > 0
        eps: numerical stabilizer
        reduction: "mean", "sum", or "none"

    Returns:
        Negative log-likelihood, reduced as requested.
    """
    mu = mu.clamp_min(eps)
    theta = theta.clamp_min(eps)

    loglik = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )
    nll = -loglik
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


# ===========================================================================
# EMA Vector Quantizer with dead-code reinitialisation
# ===========================================================================
class EMAVectorQuantizerReinit(nn.Module):
    """EMA Vector Quantizer with online dead-code reinitialisation.

    Dead codes (low EMA usage) are replaced by random encoder outputs every
    `reinit_every` steps.

    Returns (from forward):
        z_q_st        : [B, D] quantized latent (straight-through gradient)
        vq_loss       : scalar commitment loss
        usage_entropy : scalar sum(p log p), no grad
        indices        : [B] assigned code indices
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        reinit_threshold: float = 1e-3,
        reinit_every: int = 5,
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.reinit_threshold = reinit_threshold
        self.reinit_every = reinit_every
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

        embed = torch.randn(self.K, self.D)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(self.K))
        self.register_buffer("embed_avg", embed.clone())

    def _quantize(self, z: Tensor) -> Tensor:
        """Nearest-neighbour lookup via squared L2 distance."""
        z2 = torch.sum(z ** 2, dim=1, keepdim=True)          # [B, 1]
        e2 = torch.sum(self.embedding ** 2, dim=1)           # [K]
        ze = z @ self.embedding.t()                           # [B, K]
        dist = z2 - 2 * ze + e2.unsqueeze(0)                  # [B, K]
        return torch.argmin(dist, dim=1)                      # [B]

    def forward(self, z: Tensor):
        self.step += 1

        indices = self._quantize(z)
        z_q = self.embedding[indices]                         # [B, D]

        vq_loss = self.beta * F.mse_loss(z, z_q.detach())
        z_q_st = z + (z_q - z).detach()                       # straight-through

        with torch.no_grad():
            probs = self.cluster_size / (self.cluster_size.sum() + self.eps)
            usage_entropy = torch.sum(probs * torch.log(probs + self.eps))

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.K).float()
                counts = onehot.sum(0)

                self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                embed_sum = onehot.t() @ z
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(1))

                self._reinit_dead_codes(z)

        return z_q_st, vq_loss, usage_entropy, indices

    @torch.no_grad()
    def _reinit_dead_codes(self, z: Tensor):
        """Reinitialize rarely-used codes with random latent vectors."""
        if (self.step.item() % self.reinit_every) != 0:
            return

        usage = self.cluster_size / (self.cluster_size.sum() + self.eps)
        dead = usage < self.reinit_threshold
        num_dead = dead.sum().item()
        if num_dead == 0:
            return

        rand_idx = torch.randint(0, z.size(0), (num_dead,), device=z.device)
        z_rep = z[rand_idx]
        self.embedding[dead] = z_rep
        self.embed_avg[dead] = z_rep
        self.cluster_size[dead] = self.cluster_size.mean()


# ===========================================================================
# VQVAE model
# ===========================================================================
class VQVAE(nn.Module):
    """NB + EMA-VQ VQVAE with MLP encoder/decoder (vector input version).

    Each cell is represented by `num_slots` VQ codes. The encoder maps the
    input vector to a [B, num_slots * embedding_dim] latent split into
    `num_slots` independent codes quantized by the EMA-VQ layer.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_embeddings: int,
        num_slots: int = 8,
        vq_beta: float = 0.25,
        ema_decay: float = 0.99,
        vq_lambda: float = 1,
        hidden_dims: Optional[List[int]] = None,
        input_dropout: float = 0.3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.latent_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.vq_beta = vq_beta
        self.num_slots = num_slots
        self.vq_lambda = vq_lambda
        self.dropout = dropout
        self.input_dropout = input_dropout

        # Encoder MLP → [B, num_slots * embedding_dim]
        enc_modules = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_modules.append(nn.Linear(in_dim, h))
            enc_modules.append(nn.PReLU())
            enc_modules.append(nn.LayerNorm(h))
            enc_modules.append(nn.Dropout(p=self.dropout))
            in_dim = h
        enc_modules.append(nn.Linear(in_dim, num_slots * embedding_dim))
        self.encoder = nn.Sequential(*enc_modules)

        # EMA vector quantization layer
        self.vq_layer = EMAVectorQuantizerReinit(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            beta=vq_beta,
            decay=ema_decay,
            reinit_threshold=1e-3,
            reinit_every=5,
        )

        # Decoder MLP (outputs raw → mu via softplus)
        dec_modules = []
        in_dim = num_slots * embedding_dim
        dec_modules.append(nn.Dropout(p=self.input_dropout))
        for h in reversed(hidden_dims):
            dec_modules.append(nn.Linear(in_dim, h))
            dec_modules.append(nn.ReLU())
            in_dim = h
        dec_modules.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_modules)

        # Per-gene dispersion, theta = softplus(log_theta)
        self.log_theta = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to slot-wise latent vectors [B, M, D]."""
        z = self.encoder(x)  # [B, M*D]
        return z.view(z.size(0), self.num_slots, self.latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        """Decode flattened quantized latent to mean expression (>= 0)."""
        return F.softplus(self.decoder(z))

    def get_codes(self, x: Tensor) -> Tensor:
        """Return VQ code indices per cell slot, shape [B, M]."""
        z_e = self.encode(x)
        B, M, D = z_e.shape
        _, _, _, indices_flat = self.vq_layer(z_e.view(B * M, D))
        return indices_flat.view(B, M)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass.

        Returns: mu, x, vq_loss, usage_entropy, z_cell
        """
        z_e = self.encode(x)
        B, M, D = z_e.shape
        z_q_flat, vq_loss, usage_entropy, _ = self.vq_layer(z_e.view(B * M, D))
        z_q = z_q_flat.view(B, M, D)
        z_cell = z_q.reshape(B, M * D)
        mu = self.decode(z_cell)
        return mu, x, vq_loss, usage_entropy, z_cell

    def loss_function(self, *args) -> dict:
        """Compute training losses.

        Returns dict: loss, Reconstruction_Loss, VQ_Loss, Usage_Entropy, Var.
        """
        mu, x, vq_loss, usage_entropy, z_cell = args

        theta = F.softplus(self.log_theta) + 1e-4
        recons_loss = nb_nll(x, mu, theta, reduction="mean")

        z_centered = z_cell - z_cell.mean(dim=0)
        var = torch.mean(z_centered ** 2)

        return {
            "loss": recons_loss + self.vq_lambda * vq_loss,
            "Reconstruction_Loss": recons_loss,
            "VQ_Loss": vq_loss,
            "Usage_Entropy": usage_entropy,
            "Var": var,
        }

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """Generate: forward pass returning only the reconstructed mean."""
        return self.forward(x)[0]


# ===========================================================================
# Encoding / decoding helpers
# ===========================================================================
@torch.no_grad()
def encode_dataset(model: VQVAE, loader: DataLoader,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a full dataset into VQ latents and discrete codes.

    Args:
        model: trained VQVAE (eval mode).
        loader: DataLoader yielding (x,) batches.
        device: torch device.

    Returns:
        Z_e    : continuous latent, shape [N, M*D]
        Z_q    : quantized latent, shape [N, M*D]
        codes  : discrete code indices, shape [N, M]
    """
    model.eval()
    all_z_e, all_z_q, all_codes = [], [], []

    for (x,) in loader:
        z_e = model.encode(x.to(device))
        B, M, D = z_e.shape
        z_q, _, _, codes = model.vq_layer(z_e.view(B * M, D))
        all_z_e.append(z_e.cpu())
        all_z_q.append(z_q.view(B, M, D).cpu())
        all_codes.append(codes.cpu())

    N = sum(z.shape[0] for z in all_z_e)
    M = model.num_slots
    D = model.latent_dim

    Z_e = torch.cat(all_z_e, dim=0).numpy().reshape(N, M * D)
    Z_q = torch.cat(all_z_q, dim=0).numpy().reshape(N, M * D)
    codes = torch.cat(all_codes, dim=0).numpy().reshape(N, M)
    return Z_e, Z_q, codes


@torch.no_grad()
def decode_codes_to_expression(codes_tensor: Tensor, model: VQVAE) -> np.ndarray:
    """Decode VQ code sequences into mean gene-expression vectors.

    Args:
        codes_tensor: LongTensor [n_cells, num_slots].
        model: trained VQVAE.

    Returns:
        np.ndarray [n_cells, n_genes] decoded mean expression.
    """
    model.eval()
    device = next(model.parameters()).device
    codes_tensor = codes_tensor.to(device)

    embeddings = model.vq_layer.embedding[codes_tensor.view(-1)]
    embeddings = embeddings.view(codes_tensor.shape[0],
                                 codes_tensor.shape[1] * model.latent_dim)
    return model.decode(embeddings).cpu().numpy()
