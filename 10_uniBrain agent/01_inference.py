#!/usr/bin/env python3
"""
=============================================================================
File      : 01_inference.py
Directory : 10_uniBrain agent
Purpose   : Inference pipeline: VQVAE encoding + LLM cell description,
            generation, and validation.
=============================================================================
"""

import os
import re
import json
import pickle
import random
from typing import List

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    normalized_mutual_info_score, adjusted_mutual_info_score,
)
from scipy.stats import pearsonr, spearmanr

# Shared VQVAE components (see 00_vqvae_module.py)
from vqvae_module import (
    VQVAE, EMAVectorQuantizerReinit, nb_nll,
    encode_dataset, decode_codes_to_expression,
)

## Configuration ##
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS

VQVAE_CKPT = os.path.join(BASE_DIR, "data", "codebook",
                          "38w_vqvae_embdim8_numemb256_slot32_lambda03_30epoch_smallcodebook.pt")
HVG_FILE = os.path.join(BASE_DIR, "HVG.csv")
ATLAS_H5AD = os.path.join(BASE_DIR, "data", "unibrain_380k_vqvae.h5ad")
LLM_MODEL_PATH = os.path.join(BASE_DIR, "data", "LLM", "LoRA_SFT_v6_merged")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")

INPUT_DIM = 5000
EMBEDDING_DIM = 8
NUM_EMBEDDINGS = 256
NUM_SLOTS = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
DEVICE = os.environ.get("UNIBRAIN_DEVICE", "auto")
if DEVICE == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(DEVICE)


# ===========================================================================
# Part 1: Load trained VQVAE and encode cells
# ===========================================================================
model = VQVAE(
    input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM,
    num_embeddings=NUM_EMBEDDINGS, num_slots=NUM_SLOTS,
    vq_beta=0.3, ema_decay=0.99, vq_lambda=0.3,
    hidden_dims=[256, 256], dropout=0.3, input_dropout=0.4,
).to(device)

ckpt = torch.load(VQVAE_CKPT, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# Project input data to HVG space (zero-fill missing genes)
HVG_list = pd.read_csv(HVG_FILE, index_col=0).index.tolist()
adata_test = sc.read_h5ad(ATLAS_H5AD)
common_genes = [g for g in HVG_list if g in adata_test.var_names]
gene_idx = adata_test.var_names.get_indexer(common_genes)

X = adata_test.X[:, gene_idx]
X = X.toarray() if hasattr(X, "toarray") else X

X_input = np.zeros((adata_test.n_obs, len(HVG_list)))
X_input[:, [HVG_list.index(g) for g in common_genes]] = X

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_input, dtype=torch.float32)),
    batch_size=128, shuffle=False,
)

Z_e, Z_q, codes = encode_dataset(model, test_loader, device)

# Save codes and metadata for LLM tasks
test_dict = {
    'code': codes,
    'z_q': Z_q,
    'metadata': adata_test.obs[
        ['age', 'stage', 'type_2', 'region_L2', 'platform', 'sex', 'suspension_type']
    ].to_dict(orient='list'),
}
with open(os.path.join(BASE_DIR, "test_dict.pkl"), 'wb') as f:
    pickle.dump(test_dict, f)


# ===========================================================================
# Part 2: VQVAE validation
# ===========================================================================
# Reconstruction loss
total_recons_loss = 0.0
with torch.no_grad():
    for (x,) in test_loader:
        x = x.to(device)
        mu, x_in, vq_loss, _, _ = model(x)
        theta = F.softplus(model.log_theta) + 1e-4
        total_recons_loss += nb_nll(x_in, mu, theta, reduction="mean").item() * x.size(0)
avg_recons_loss = total_recons_loss / len(test_loader.dataset)
print(f"VQVAE reconstruction loss (NB-NLL): {avg_recons_loss:.4f}")

# Cell-type classification from quantized latent
X_train, X_test, y_train, y_test = train_test_split(
    Z_q, test_dict["metadata"]['type_2'], test_size=0.2, random_state=42,
)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Cell-type classification -> Accuracy: {accuracy_score(y_test, y_pred):.4f}, "
      f"Macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

# Dead-code ratio
usage = model.vq_layer.cluster_size / (model.vq_layer.cluster_size.sum() + 1e-5)
dead_codes = (usage < model.vq_layer.reinit_threshold).sum().item()
print(f"VQ dead-code ratio: {dead_codes / model.vq_layer.K:.4f}")


# ===========================================================================
# Part 3: Cell description test with fine-tuned LLM
# ===========================================================================
def codes_to_text(codes: np.ndarray) -> str:
    """Convert a cell's VQ code sequence into a text token string."""
    parts = ["<CELL_START>"] + [f"<CODE {c}>" for c in codes] + ["<CELL_END>"]
    return "".join(parts)


# Build cell-description test corpus
test_prompts_list = [
    {"instruction": f"{codes_to_text(test_dict['code'][i, :])}, Describe the cell:",
     "input": "", "output": ""}
    for i in range(test_dict['code'].shape[0])
]
with open(os.path.join(CORPUS_DIR, "cell_380k_describe_test_v6.json"), 'w') as f:
    json.dump(test_prompts_list, f)

# Load fine-tuned Qwen3 LLM
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

LLM = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)


def generate_response(prompt: str, max_new_tokens: int = 256,
                      do_sample: bool = False, temperature: float = 0.8,
                      top_p: float = 0.9, repetition_penalty: float = 1.1) -> str:
    """Generate a single LLM response for the given prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = LLM.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=False)


# Sample balanced test subset and generate region predictions
random.seed(42)
np.random.seed(42)
total_cells = len(test_dict["metadata"]['type_2'])
test_indices = sorted(random.sample(range(total_cells), min(1000, total_cells)))
test_prompts = [test_prompts_list[i] for i in test_indices]

result_list = [generate_response(p['instruction']) for p in tqdm(test_prompts, desc="Cell description")]

labels = [test_dict["metadata"]["region_L2"][i] for i in test_indices]
preds = [r.split(" from the ")[1].split(" of a ")[0].rstrip(".").strip() for r in result_list]

print(f"Macro-F1: {f1_score(labels, preds, average='macro'):.4f}")
print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
print(classification_report(labels, preds, zero_division=0))

all_classes = list(set(labels).union(set(preds)))
cm = pd.DataFrame(
    confusion_matrix(labels, preds, labels=all_classes),
    index=all_classes, columns=all_classes,
)
(cm / cm.sum(axis=0)).to_csv(os.path.join(OUTPUT_DIR, "prediction_proportion.csv"))


# ===========================================================================
# Part 4: Cell generation (codes from text, expression from codes)
# ===========================================================================
def generate_cell_codes_from_corpus(cell_generation_corpus: List[dict],
                                    generate_func, num_slots: int = NUM_SLOTS,
                                    num_embeddings: int = NUM_EMBEDDINGS) -> torch.Tensor:
    """Generate VQ code sequences from a cell-generation corpus using the LLM."""
    all_codes = []
    for item in tqdm(cell_generation_corpus):
        code_str = generate_func(item['instruction'])
        matches = re.findall(r'<CODE\s+(\d+)>', code_str)
        codes_list = [int(m) for m in matches if 0 <= int(m) < num_embeddings]

        if not codes_list:
            codes_list = [0]
        if len(codes_list) > num_slots:
            codes_list = codes_list[:num_slots]
        elif len(codes_list) < num_slots:
            codes_list = codes_list + [codes_list[-1]] * (num_slots - len(codes_list))
        all_codes.append(codes_list)
    return torch.tensor(all_codes, dtype=torch.long)


def create_anndata_from_generated_cells(expression_matrix: np.ndarray,
                                        original_adata: ad.AnnData) -> ad.AnnData:
    """Wrap a generated expression matrix into an AnnData object."""
    n_cells = expression_matrix.shape[0]
    gene_names = original_adata.var_names.tolist()[:model.input_dim]
    obs_df = original_adata.obs.iloc[:n_cells].copy()
    obs_df['generated'] = True
    obs_df['source'] = 'llm_generated'
    return ad.AnnData(X=expression_matrix, obs=obs_df, var=pd.DataFrame(index=gene_names))


def meta_to_text(meta) -> str:
    """Build a textual cell description from cell metadata."""
    meta_str = f"This is a {meta['type_2']} from the {meta['region_L2']} of a "
    if meta['sex'] == 'unknown':
        meta_str += f"{meta['stage'].split('-')[1]} person."
    else:
        meta_str += f"{meta['stage'].split('-')[1]} {meta['sex']}."
    suspension = "single-cell" if meta['suspension_type'] == 'sc' else "single-nucleus"
    meta_str += f"The cell is sequenced by {meta['platform']}-{suspension} platform."
    return meta_str


# Generation test corpus (hs33, 5000 cells)
adata = sc.read_h5ad(ATLAS_H5AD)
adata_hs33 = adata[adata.obs.dataset == "hs33"]
adata_hs33 = adata_hs33[~adata_hs33.obs.type_2.isin(["Thalamic_GABA", "Cholinergic"])]
adata_hs33 = sc.pp.subsample(adata_hs33, n_obs=5000, random_state=123, copy=True)

cell_generation_corpus = [
    {"instruction": f"Generate a cell. {meta_to_text(adata_hs33.obs.iloc[i])}",
     "input": "", "output": ""}
    for i in range(adata_hs33.n_obs)
]
with open(os.path.join(CORPUS_DIR, "cell_generation_test.json"), 'w') as f:
    json.dump(cell_generation_corpus, f)

codes_tensor = generate_cell_codes_from_corpus(cell_generation_corpus, generate_response)
expression_matrix = decode_codes_to_expression(codes_tensor, model)
generated_adata = create_anndata_from_generated_cells(expression_matrix, adata_hs33)
generated_adata.obsm["X_vq"] = codes_tensor.cpu().numpy()
generated_adata = generated_adata[
    ~generated_adata.obs["type_2"].isin(
        ["Remove", "Hypothalamic_Glu", "Midbrain-derived_GABA", "Neuronal_IPC"]
    )
]
generated_adata.write_h5ad(os.path.join(OUTPUT_DIR, "v6.2_model_hs33_generation.h5ad"))


# ===========================================================================
# Part 5: Validate generated cells against real data
# ===========================================================================
ad_hs33 = sc.read_h5ad(ATLAS_H5AD)
ad_hs33 = ad_hs33[ad_hs33.obs.dataset == "hs33"]
ad_hs33 = sc.pp.subsample(ad_hs33, n_obs=5000, random_state=123, copy=True)
ad_hs33.obs['generated'] = False

combined_data = ad.concat([ad_hs33, generated_adata], join='inner')
combined_data.obs["generated"] = combined_data.obs["generated"].astype("category")

# Expression distribution comparison
real_mask = (combined_data.obs["generated"] == False).values  # noqa: E712
gen_mask = (combined_data.obs["generated"] == True).values    # noqa: E712

X_c = combined_data.X.toarray() if hasattr(combined_data.X, "toarray") else combined_data.X
X_real = X_c[real_mask].flatten()
X_gen = X_c[gen_mask].flatten()
X_real, X_gen = X_real[np.isfinite(X_real)], X_gen[np.isfinite(X_gen)]

n = min(X_real.shape[0], X_gen.shape[0])
idx, idy = np.random.choice(X_real.shape[0], n, replace=False), np.random.choice(X_gen.shape[0], n, replace=False)

pcc, _ = pearsonr(X_real[idx], X_gen[idy])
scc, _ = spearmanr(X_real[idx], X_gen[idy])
ed = np.linalg.norm(X_real[idx] - X_gen[idy]) / n
print(f"Real vs generated -> PCC: {pcc:.4f}, SCC: {scc:.4f}, ED: {ed:.4f}")

# Cluster-structure comparison
sc.pp.highly_variable_genes(combined_data, n_top_genes=1000, flavor="seurat", subset=True)
sc.tl.pca(combined_data, n_comps=30, svd_solver='arpack')
sc.pp.neighbors(combined_data, n_neighbors=15, n_pcs=30)
sc.tl.leiden(combined_data, resolution=0.02)

for name, mask in [("all", np.ones(combined_data.n_obs, dtype=bool)),
                   ("real", real_mask), ("generated", gen_mask)]:
    nmi = normalized_mutual_info_score(
        combined_data.obs.loc[mask, "type_2"], combined_data.obs.loc[mask, "leiden"])
    ami = adjusted_mutual_info_score(
        combined_data.obs.loc[mask, "type_2"], combined_data.obs.loc[mask, "leiden"])
    print(f"NMI ({name}): {nmi:.4f}, AMI ({name}): {ami:.4f}")

# Reconstruction quality of real hs33 data
X_arr = ad_hs33.X.toarray() if hasattr(ad_hs33.X, "toarray") else ad_hs33.X
X_arr = torch.tensor(X_arr, dtype=torch.float32)
recon_loader = DataLoader(TensorDataset(X_arr), batch_size=128, shuffle=False)

Z_e_recon, Z_q_recon, _ = encode_dataset(model, recon_loader, device)
X_recon = model.decode(torch.tensor(Z_q_recon, dtype=torch.float32).to(device)).cpu().numpy()
recon_adata = ad.AnnData(X=X_recon, obs=ad_hs33.obs, var=ad_hs33.var)
recon_adata.write_h5ad(os.path.join(OUTPUT_DIR, "v6_model_hs33_reconstruct.h5ad"))

X_real_flat, recon_flat = X_arr.numpy().flatten(), X_recon.flatten()
pcc_r, _ = pearsonr(X_real_flat, recon_flat)
scc_r, _ = spearmanr(X_real_flat, recon_flat)
ed_r = np.linalg.norm(X_real_flat - recon_flat) / len(X_real_flat)
print(f"VQVAE reconstruction -> PCC: {pcc_r:.4f}, SCC: {scc_r:.4f}, ED: {ed_r:.4f}")

pcc_per_cell = [pearsonr(X_arr[i].numpy(), X_recon[i])[0] for i in range(len(X_arr))]
print(f"Mean per-cell reconstruction PCC: {np.mean(pcc_per_cell):.4f}")


# ===========================================================================
# Part 6: Microglia lifespan generation (15 stages × 100 cells)
# ===========================================================================
STAGES = [
    '01-Early first trimester', '02-Late first trimester',
    '03-Early second trimester', '04-Mid second trimester',
    '05-Late second trimester', '06-Early third trimester',
    '07-Late third trimester', '08-Neonatal and early infancy',
    '09-Late infancy', '10-Early childhood', '11-Middle and late childhood',
    '12-Adolescence', '13-Young adulthood', '14-Middle adulthood', '15-Late adulthood',
]

microglia_corpus = []
for stage in tqdm(STAGES):
    for _ in range(100):
        meta = {"type_2": "Microglia", "region_L2": "Hippocampus", "stage": stage,
                "sex": "unknown", "platform": "10X",
                "suspension_type": random.choice(["sc", "sn"])}
        microglia_corpus.append({
            "instruction": f"Generate a cell. {meta_to_text(meta)}",
            "input": "", "output": "",
        })

mg_codes = generate_cell_codes_from_corpus(microglia_corpus, generate_response, num_slots=32)
mg_expression = decode_codes_to_expression(mg_codes, model)


def parse_instruction_to_obs(corpus, stage_labels=None, region_labels=None,
                             celltype_labels=None, default_type=None):
    """Parse generated-instruction strings back into cell metadata."""
    obs_list = []
    for item in corpus:
        instruction = item["instruction"]
        cur_stage = next((s for s in (stage_labels or [])
                          if s.split("-", 1)[1] in instruction), None)
        cur_region = next((r for r in (region_labels or [])
                           if f"from the {r} of" in instruction), None)
        cur_celltype = default_type
        if celltype_labels:
            cur_celltype = next((ct for ct in celltype_labels
                                 if f"This is a {ct} from" in instruction), default_type)
        sex = "female" if " female." in instruction else ("male" if " male." in instruction else "unknown")
        suspension_type = "sc" if "single-cell" in instruction else "sn"
        obs_list.append({
            "type_2": cur_celltype, "region_L2": cur_region, "stage": cur_stage,
            "sex": sex, "platform": "10X", "suspension_type": suspension_type,
            "generated": True,
        })
    return obs_list


gene_names = pd.read_csv(os.path.join(OUTPUT_DIR, "generation_gene_list.txt"),
                         header=None)[0].tolist()

mg_obs = pd.DataFrame(parse_instruction_to_obs(
    microglia_corpus, stage_labels=STAGES, default_type="Microglia"))
mg_obs.index = [f"gen_cell_{i}" for i in range(len(mg_obs))]

mg_adata = ad.AnnData(X=mg_expression, obs=mg_obs, var=pd.DataFrame(index=gene_names))
mg_adata.uns["description"] = "Generated hippocampal microglia across lifespan"
mg_adata.uns["model"] = "llm v6.2"

phase_map = {
    "01-Early first trimester": "First trimester", "02-Late first trimester": "First trimester",
    "03-Early second trimester": "Second trimester", "04-Mid second trimester": "Second trimester",
    "05-Late second trimester": "Second trimester", "06-Early third trimester": "Third trimester",
    "07-Late third trimester": "Third trimester", "08-Neonatal and early infancy": "Infancy",
    "09-Late infancy": "Infancy", "10-Early childhood": "Childhood",
    "11-Middle and late childhood": "Childhood", "12-Adolescence": "Adolescence",
    "13-Young adulthood": "Adulthood", "14-Middle adulthood": "Adulthood",
    "15-Late adulthood": "Aged",
}
mg_adata.obs["phase"] = mg_adata.obs["stage"].map(phase_map)
mg_adata.write_h5ad(os.path.join(OUTPUT_DIR, "microglia_lifespan_generation.h5ad"))


# ===========================================================================
# Part 7: Full brain atlas generation (15 × 24 × 38 combinations)
# ===========================================================================
REGIONS = [
    "Basal forebrain", "Basal ganglia", "Brain", "Cerebellum", "Cerebral cortex",
    "Cerebrum", "Diencephalon", "Epithalamus", "Frontal lobe", "Hippocampus",
    "Hypothalamus", "Insular lobe", "Medulla", "Mesencephalon", "Occipital lobe",
    "Parietal lobe", "Pituitary gland", "Pons", "Prosencephalon", "Rhombencephalon",
    "Spinal cord", "Telencephalon", "Temporal lobe", "Thalamus",
]
CELLTYPES = [
    "Astrocyte", "CALB2+ GABA", "Cholinergic", "Choroid_plexus_epithelial",
    "Endothelial_cell", "Ependymal", "Fibroblast", "Glioblast", "Granule_neuron",
    "Hippocampal_Glu", "Hypothalamic_Glu", "L2/3-CUX2 Glu", "L2/3-SATB2 Glu",
    "L4-RORB Glu", "L5-BCL11B Glu", "L5/6-THEMIS Glu", "L5/6-TLE4 Glu",
    "LHX6+LAMP5+ GABA", "Macrophage", "Medium_spiny_neuron", "Microglia",
    "Midbrain-derived_GABA", "Neuroblast", "Neuronal_IPC", "Oligodendrocyte",
    "Oligodendrocyte_precursor", "Oligodendrocyte_precursor_committed", "Pericyte",
    "Purkinje_neuron", "PVALB+ GABA", "Radial_glia", "RELN+ GABA", "Serotonergic",
    "SST+ GABA", "Thalamic_GABA", "Thalamic_Glu",
    "Vascular_smooth_muscle_cell", "VIP+ GABA",
]

import itertools  # noqa: E402

atlas_corpus = []
for stage, region, celltype in itertools.product(STAGES, REGIONS, CELLTYPES):
    meta = {"type_2": celltype, "region_L2": region, "stage": stage,
            "sex": "unknown", "platform": "10X", "suspension_type": "sc"}
    atlas_corpus.append({
        "instruction": f"Generate a cell. {meta_to_text(meta)}",
        "input": "", "output": "",
    })
print(f"Total generation tasks: {len(atlas_corpus)} (15 × 24 × 38)")

atlas_codes = generate_cell_codes_from_corpus(atlas_corpus, generate_response, num_slots=32)
atlas_expression = decode_codes_to_expression(atlas_codes, model)

atlas_obs = pd.DataFrame(parse_instruction_to_obs(
    atlas_corpus, stage_labels=STAGES, region_labels=REGIONS, celltype_labels=CELLTYPES))
atlas_obs.index = [f"gen_cell_{i}" for i in range(len(atlas_obs))]

atlas_adata = ad.AnnData(X=atlas_expression, obs=atlas_obs, var=pd.DataFrame(index=gene_names))
atlas_adata.uns["description"] = "Generated brain atlas across stages, regions and cell types"
atlas_adata.uns["model"] = "llm v6.2"
atlas_adata.obs["phase"] = atlas_adata.obs["stage"].map(phase_map)
atlas_adata.write_h5ad(os.path.join(OUTPUT_DIR, "generated_brain_atlas.h5ad"))
