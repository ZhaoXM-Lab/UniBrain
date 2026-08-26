#!/usr/bin/env python3
"""
=============================================================================
File      : 04_update_tokenizer.py
Directory : 10_uniBrain agent
Purpose   : Extend the Qwen3 tokenizer with VQ code and cell boundary special tokens.
=============================================================================
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

## Configuration ##
# --- Set these paths for your own environment ---
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # project root <-- MODIFY THIS
MODEL_DIR = os.path.join(BASE_DIR, "result", "models")          # output dir
LLM_DIR = os.path.join(BASE_DIR, "LLM")                         # base LLM files

# Path to the base Qwen3 model (used to read the tokenizer and embeddings)
MODEL_NAME_OR_PATH = os.path.join(LLM_DIR, "Qwen3-8B")

# Output paths: updated tokenizer + resized model
TOKENIZER_OUT_DIR = os.path.join(MODEL_DIR, "Qwen3-8B_update")
MODEL_OUT_DIR = os.path.join(MODEL_DIR, "Qwen3-8B_update")

# Number of VQ code tokens. MUST equal the VQVAE codebook size (NUM_EMBEDDINGS
# in 02_vqvae.py, e.g. 256) so that every code id in the corpus maps to a token.
NUM_CODE_TOKENS = 256

os.makedirs(TOKENIZER_OUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)


def build_new_tokens(num_code_tokens: int) -> list[str]:
    """Build the list of special tokens for VQ code sequences."""
    new_tokens = [f"<CODE {i}>" for i in range(num_code_tokens)]
    new_tokens.append("<CELL_START>")
    new_tokens.append("<CELL_END>")
    return new_tokens


new_tokens = build_new_tokens(NUM_CODE_TOKENS)
print(f"Adding {len(new_tokens)} new special tokens "
      f"(<CODE 0>..<CODE {NUM_CODE_TOKENS - 1}>, <CELL_START>, <CELL_END>)")

# ---------------------------------------------------------------------------
# Load tokenizer and add the new special tokens
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

# Only tokens not already present are added (idempotent on re-run)
added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
print(f"Added {added} new tokens to the vocabulary")

# --- Sanity check: a sample cell sequence must tokenize into single units ---
sample = ("<CELL_START><CODE 32><CODE 18><CODE 51><CODE 26>"
          "<CODE 3><CODE 24><CODE 16><CODE 45><CODE 12><CODE 55>"
          "<CODE 46><CODE 2><CODE 33><CODE 3><CODE 46><CODE 19>"
          "<CODE 31><CODE 40><CODE 32><CODE 33><CODE 15><CODE 8>"
          "<CODE 17><CODE 32><CODE 11><CODE 19><CODE 29><CODE 29>"
          "<CODE 61><CODE 31><CODE 41><CODE 46><CELL_END>")
encoded = tokenizer.encode(sample)
print(f"Sanity check: {len(sample.split('>')) - 1} tokens -> "
      f"{len(encoded)} ids (should be equal: {len(sample.split('>')) - 1 == len(encoded)})")

# Save the updated tokenizer
tokenizer.save_pretrained(TOKENIZER_OUT_DIR)
print(f"Tokenizer saved to {TOKENIZER_OUT_DIR}")

# ---------------------------------------------------------------------------
# Load the base model, resize embeddings to the new vocabulary, and save
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    low_cpu_mem_usage=True,
    device_map=None,
    trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))
print(f"Embedding matrix resized to {model.get_input_embeddings().num_embeddings}")

model.save_pretrained(MODEL_OUT_DIR)
print(f"Model saved to {MODEL_OUT_DIR}")
print("Done.")
