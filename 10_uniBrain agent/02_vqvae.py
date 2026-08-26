#!/usr/bin/env python3
"""
=============================================================================
File      : 02_vqvae.py
Directory : 10_uniBrain agent
Purpose   : Train a Negative-Binomial VQVAE to encode single-cell expression
            into discrete code tokens, then encode the full dataset and
            optional held-out test sets.
=============================================================================
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# Shared VQVAE components (see 00_vqvae_module.py)
from vqvae_module import VQVAE, encode_dataset

## Configuration ##
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "result", "models", "vqvae")

INPUT_H5AD = os.path.join(DATA_DIR, "unibrain_380k_vqvae.h5ad")
HVG_TXT = os.path.join(DATA_DIR, "highly_variable_genes_5000.txt")
OUTPUT_H5AD = os.path.join(DATA_DIR, "unibrain_380k_vqvae_smallcodebook_p8.h5ad")

# Model / training hyper-parameters
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
EMBEDDING_DIM = 8
NUM_EMBEDDINGS = 256
NUM_SLOTS = 32
VQ_BETA = 0.3
EMA_DECAY = 0.99
VQ_LAMBDA = 0.3
HIDDEN_DIMS = [256, 256]
DROPOUT = 0.3
INPUT_DROPOUT = 0.4

# Optional: apply trained model to held-out test datasets
APPLY_TO_TEST = True
TEST_H5AD = os.path.join(DATA_DIR, "hs_norm.h5ad")
TEST_DATASETS = ["hs20", "hs29"]
TEST_OUTPUT_H5AD = os.path.join(DATA_DIR, "unibrain_380k_test_vqvae_smallcodebook_p8.h5ad")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================================================================
# Data loading
# ===========================================================================
print("Loading data ...")
adata = sc.read_h5ad(INPUT_H5AD)
print(f"Loaded AnnData: {adata.n_obs} cells x {adata.n_vars} genes")

X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
X = torch.tensor(X, dtype=torch.float32)

train_data = TensorDataset(X)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
DEVICE = os.environ.get("UNIBRAIN_DEVICE", "auto")
if DEVICE == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(DEVICE)

# ===========================================================================
# Model setup
# ===========================================================================
model = VQVAE(
    input_dim=X.shape[1], embedding_dim=EMBEDDING_DIM,
    num_embeddings=NUM_EMBEDDINGS, num_slots=NUM_SLOTS,
    vq_beta=VQ_BETA, ema_decay=EMA_DECAY, vq_lambda=VQ_LAMBDA,
    hidden_dims=HIDDEN_DIMS, dropout=DROPOUT, input_dropout=INPUT_DROPOUT,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=5e-5)

# ===========================================================================
# Training loop
# ===========================================================================
print(f"Training on device: {device}")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = total_recon = total_vq = total_usage_ent = total_var = 0.0

    for (data,) in train_loader:
        mu, x, vq_loss, usage_entropy, z_cell = model(data.to(device))
        losses = model.loss_function(mu, x, vq_loss, usage_entropy, z_cell)

        losses["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += losses["loss"].item()
        total_recon += losses["Reconstruction_Loss"].item()
        total_vq += losses["VQ_Loss"].item()
        total_usage_ent += losses["Usage_Entropy"].item()
        total_var += losses["Var"].item()

    scheduler.step()
    n = len(train_loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {total_loss/n:.4f} | "
          f"NB_NLL: {total_recon/n:.4f} | VQ: {total_vq/n:.4f} | "
          f"UsageEnt: {total_usage_ent/n:.4f} | Var: {total_var/n:.4f}")

# --- Save checkpoint ---
ckpt_name = (f"38w_vqvae_embdim{EMBEDDING_DIM}_numemb{NUM_EMBEDDINGS}_"
             f"slot{NUM_SLOTS}_lambda{VQ_LAMBDA}_{NUM_EPOCHS}epoch_smallcodebook.pt")
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": NUM_EPOCHS}, os.path.join(MODEL_DIR, ckpt_name))
print(f"Checkpoint saved to {os.path.join(MODEL_DIR, ckpt_name)}")

# ===========================================================================
# Encode full dataset and save
# ===========================================================================
print("Encoding all cells ...")
test_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
Z_e, Z_q, codes = encode_dataset(model, test_loader, device)

# Codebook usage summary
unique_rows, counts = np.unique(codes, axis=0, return_counts=True)
print(f"Unique code sequences: {len(unique_rows)}, max reuse: {max(counts)}")

adata.obsm["X_vqvae_continuous"] = Z_e
adata.obsm["X_vqvae_discrete"] = Z_q
adata.uns["vqvae_codes"] = codes
adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
print(f"Encoded AnnData saved to {OUTPUT_H5AD}")

# ===========================================================================
# (Optional) Encode held-out test datasets
# ===========================================================================
if APPLY_TO_TEST:
    print("Applying model to test datasets ...")

    HVG_list = pd.read_table(HVG_TXT, header=None)[0].tolist()
    adata_test = sc.read_h5ad(TEST_H5AD)
    adata_test = adata_test[adata_test.obs["dataset"].isin(TEST_DATASETS)]
    adata_test = adata_test[:, HVG_list]

    X_test = adata_test.X.toarray() if hasattr(adata_test.X, "toarray") else adata_test.X
    X_test = torch.tensor(X_test, dtype=torch.float32)
    test_loader2 = DataLoader(TensorDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)

    Z_e_t, Z_q_t, codes_t = encode_dataset(model, test_loader2, device)

    adata_test.obsm["X_vqvae_continuous"] = Z_e_t
    adata_test.obsm["X_vqvae_discrete"] = Z_q_t
    adata_test.uns["vqvae_codes"] = codes_t
    adata_test.write_h5ad(TEST_OUTPUT_H5AD, compression="gzip")
    print(f"Encoded test AnnData saved to {TEST_OUTPUT_H5AD}")

print("Done.")
