#!/usr/bin/env python3
"""
=============================================================================
File      : 03_corpus_construction.py
Directory : 10_uniBrain agent
Purpose   : Build the SFT text corpus (cell-type, brain-region, single-cell and merged corpora) for LLM fine-tuning.
=============================================================================
"""

import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

## Configuration ##
# --- Set these paths for your own environment ---
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # project root <-- MODIFY THIS
DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_DIR = os.path.join(BASE_DIR, "result", "corpus")          # output corpus files
DESC_DIR = os.path.join(BASE_DIR, "data", "descriptions")        # *_description.json files
GENERAL_CORPUS_DIR = os.path.join(BASE_DIR, "data", "general_corpus")  # external corpora

# Input AnnData produced by 02_vqvae.py (contains vqvae_codes in uns)
INPUT_H5AD = os.path.join(DATA_DIR, "unibrain_380k_vqvae_smallcodebook_p8.h5ad")

# Description files (pre-written knowledge text)
CELLTYPE_DESC = os.path.join(DESC_DIR, "cell_type_description.json")
BRAIN_REGION_DESC = os.path.join(DESC_DIR, "brain_region_L2_description.json")

# General / common-knowledge corpora (instruction-format JSON files)
MEDICAL_WIKI = os.path.join(GENERAL_CORPUS_DIR, "medical_wiki_ft.json")
ALPACA_GPT4 = os.path.join(GENERAL_CORPUS_DIR, "alpaca_gpt4_data_en.json")
PUBMEDQA_TRAIN = os.path.join(GENERAL_CORPUS_DIR, "PubMedQA", "train.json")
PUBMEDQA_TEST = os.path.join(GENERAL_CORPUS_DIR, "PubMedQA", "test.json")

# Oversampling factors used when merging the corpora
COMMON_CORPUS_REPEAT = 20      # repeat general corpora
BRAIN_REGION_REPEAT = 100      # repeat brain-region corpus
CELLTYPE_REPEAT = 100          # repeat cell-type corpus

os.makedirs(CORPUS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Cell-type knowledge corpus
# ---------------------------------------------------------------------------
def build_cell_type_corpus(celltype_info: pd.DataFrame) -> list[dict]:
    """Build cell-type Q&A samples from the cell-type description table.

    The table has columns: tree_L1 (major class), tree_L2 (subclass) and
    description, indexed by the cell subtype name.
    """
    corpus = []
    for subtype in celltype_info.index:
        desc = celltype_info.loc[subtype, "description"]
        corpus.append({
            "instruction": f"Tell me the information of the cell type {subtype}.",
            "input": "",
            "output": desc,
            "history": [],
        })
        corpus.append({
            "instruction": f"What is a {subtype}?",
            "input": "",
            "output": desc,
            "history": [],
        })

    # Major cell classes of the human brain
    corpus.append({
        "instruction": "What are the major cell classes in the human brain?",
        "input": "",
        "output": "The major cell classes in the human brain including "
                  f"{', '.join(celltype_info.tree_L1.unique())}.",
        "history": [],
    })

    # Subtypes of each major class (tree_L1)
    for L1 in celltype_info.tree_L1.unique():
        corpus.append({
            "instruction": f"What are the major subtypes of {L1} in the human brain?",
            "input": "",
            "output": f"The subtypes of {L1} including "
                      f"{', '.join(celltype_info[celltype_info.tree_L1 == L1].tree_L2.unique())}.",
            "history": [],
        })

    # Subtypes of each subclass (tree_L2)
    for L2 in celltype_info.tree_L2.unique():
        corpus.append({
            "instruction": f"What are the major subtypes of {L2} in the human brain?",
            "input": "",
            "output": f"The subtypes of {L2} including "
                      f"{', '.join(celltype_info[celltype_info.tree_L2 == L2].index)}.",
            "history": [],
        })

    # Which major class a subtype belongs to
    for subtype in celltype_info.index:
        corpus.append({
            "instruction": f"Which major cell type does {subtype} belong to?",
            "input": "",
            "output": f"{subtype} is a subtype of "
                      f"{celltype_info.loc[subtype, 'tree_L2']}, which also belongs to "
                      f"the major cell type {celltype_info.loc[subtype, 'tree_L1']}.",
            "history": [],
        })

    # Major class of each subclass (tree_L2)
    for L2 in celltype_info.tree_L2.unique():
        corpus.append({
            "instruction": f"What is the major cell type of {L2} in the human brain?",
            "input": "",
            "output": f"{L2} is a subtype of "
                      f"{celltype_info.loc[celltype_info.tree_L2 == L2, 'tree_L1'].iloc[0]}.",
            "history": [],
        })

    return corpus


with open(CELLTYPE_DESC, "r") as f:
    celltype_info = pd.DataFrame(json.load(f)).transpose()

cell_type_corpus = build_cell_type_corpus(celltype_info)
print(f"Cell-type corpus size: {len(cell_type_corpus)}")

with open(os.path.join(CORPUS_DIR, "cell_type.json"), "w") as f:
    json.dump(cell_type_corpus, f)


# ---------------------------------------------------------------------------
# 2. Brain-region knowledge corpus
# ---------------------------------------------------------------------------
def build_brain_region_corpus(brain_region_info: dict, adata) -> list[dict]:
    """Build brain-region Q&A samples from the region description dictionary
    and the region hierarchy stored in the AnnData obs columns."""
    corpus = []
    for region, info in brain_region_info.items():
        corpus.append({
            "instruction": f"What is {region}?",
            "input": "",
            "output": info["description"],
            "history": [],
        })
        corpus.append({
            "instruction": f"Please describe the {region} in the human brain.",
            "input": "",
            "output": info["description"],
            "history": [],
        })

    # All major brain regions
    corpus.append({
        "instruction": "What are the major brain regions in the human brain?",
        "input": "",
        "output": "The major brain regions in the human brain including "
                  f"{', '.join(brain_region_info.keys())}.",
        "history": [],
    })

    # Functional sub-structures of each region (region_L3 of that region_L2)
    for region in brain_region_info.keys():
        sub_regions = adata.obs["region_L3"][
            adata.obs["region_L2"] == region].astype(str).unique()
        corpus.append({
            "instruction": f"What functional structures belong to {region}?",
            "input": "",
            "output": f"The functional structures of {region} including "
                      f"{', '.join(sub_regions)}.",
            "history": [],
        })

    return corpus


with open(BRAIN_REGION_DESC, "r") as f:
    brain_region_info = json.load(f)

adata = sc.read_h5ad(INPUT_H5AD)
brain_region_corpus = build_brain_region_corpus(brain_region_info, adata)
print(f"Brain-region corpus size: {len(brain_region_corpus)}")

with open(os.path.join(CORPUS_DIR, "brain_region.json"), "w") as f:
    json.dump(brain_region_corpus, f)


# ---------------------------------------------------------------------------
# 3. Single-cell corpus (VQ code sequence <-> cell description)
# ---------------------------------------------------------------------------
# Rebuild suspension_type / platform from the dataset id (public dataset IDs)
seq_map = {
    "sn": ["hs01", "hs02", "hs03", "hs04", "hs05", "hs06", "hs07", "hs08",
           "hs13", "hs15", "hs20", "hs21", "hs23", "hs26", "hs27", "hs28",
           "hs29", "hs31", "hs33", "hs34", "hs40", "hs41", "hs42", "hs44"],
    "sc": ["hs09", "hs11", "hs12", "hs14", "hs16", "hs17", "hs18", "hs19",
           "hs22", "hs24", "hs25", "hs30", "hs32", "hs35", "hs36", "hs37",
           "hs38", "hs39", "hs43"],
}
adata.obs["suspension_type"] = adata.obs["dataset"].map(
    lambda x: "sn" if x in seq_map["sn"] else "sc")

plat_map = {
    "10X": ["hs01", "hs02", "hs03", "hs04", "hs05", "hs07", "hs08", "hs13",
            "hs16", "hs17", "hs18", "hs19", "hs20", "hs21", "hs22", "hs23",
            "hs24", "hs25", "hs26", "hs27", "hs28", "hs29", "hs30", "hs31",
            "hs32", "hs33", "hs34", "hs36", "hs37", "hs39", "hs42", "hs44"],
    "Drop-seq": ["hs06"],
    "STRT-seq": ["hs09", "hs11", "hs38"],
    "Fluidigm C1": ["hs14", "hs43"],
    "SPLiT-seq": ["hs15", "hs35", "hs41"],
    "Smart-seq": ["hs40"],
}


def _map_platform(dataset: str) -> str:
    for platform, datasets in plat_map.items():
        if dataset in datasets:
            return platform
    return "Smart-seq"  # fallback


adata.obs["platform"] = adata.obs["dataset"].map(_map_platform)


def codes_to_text(codes) -> str:
    """Convert one cell's VQ code sequence into token text."""
    codes_str = "<CELL_START>"
    for code in codes:
        codes_str += f"<CODE {code}>"
    codes_str += "<CELL_END>"
    return codes_str.strip()


def meta_to_text(meta) -> str:
    """Convert cell metadata into a natural-language description."""
    meta_str = "This is a "
    meta_str += f"{meta['type_2']} from the {meta['region_L2']} of a "
    if meta["sex"] == "unknown":
        meta_str += f"{meta['stage'].split('-')[1]} person."
    else:
        meta_str += f"{meta['stage'].split('-')[1]} {meta['sex']}."
    if meta["suspension_type"] == "sc":
        suspension = "single-cell"
    else:
        suspension = "single-nucleus"
    meta_str += f"The cell is sequenced by {meta['platform']}-{suspension} platform."
    return meta_str


cell_description_corpus = []
for i in tqdm(range(adata.n_obs), desc="Building single-cell corpus"):
    code_str = codes_to_text(adata.uns["vqvae_codes"][i])
    cell_meta = adata.obs.iloc[i]
    cell_description = meta_to_text(adata.obs.iloc[i])

    # whole-cell description
    cell_description_corpus.append({
        "instruction": f"{code_str}, Describe the cell:",
        "input": "",
        "output": cell_description,
        "history": [],
    })
    # cell type
    cell_description_corpus.append({
        "instruction": f"{code_str}, what is the cell type of this cell?",
        "input": "",
        "output": f"This cell is a {cell_meta['type_2']}.",
        "history": [],
    })
    # platform
    cell_description_corpus.append({
        "instruction": f"{code_str}, which platform is this cell sequenced by?",
        "input": "",
        "output": (f"This cell is sequenced by {cell_meta['platform']}-"
                   f"{cell_meta['suspension_type']} platform."),
        "history": [],
    })
    # region
    cell_description_corpus.append({
        "instruction": f"{code_str}, which brain region is this cell from?",
        "input": "",
        "output": f"This cell is from the {cell_meta['region_L2']}.",
        "history": [],
    })
    # stage
    cell_description_corpus.append({
        "instruction": f"{code_str}, which development stage is this cell from?",
        "input": "",
        "output": f"This cell is from the {cell_meta['stage']}.",
        "history": [],
    })
    # generation task (description -> code sequence)
    cell_description_corpus.append({
        "instruction": f"Generate a cell. {cell_description}",
        "input": "",
        "output": code_str,
        "history": [],
    })

print(f"Single-cell corpus size: {len(cell_description_corpus)}")

with open(os.path.join(CORPUS_DIR, "cell_task_v6.json"), "w") as f:
    json.dump(cell_description_corpus, f)


# ---------------------------------------------------------------------------
# 4. Merge with common-knowledge corpora
# ---------------------------------------------------------------------------
def load_json(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# Drop the unused "history" field for the final merged dataset
for corpus in [cell_description_corpus, brain_region_corpus, cell_type_corpus]:
    for sample in corpus:
        del sample["history"]

# General corpora (kept as-is, only oversampled)
medical_wiki = load_json(MEDICAL_WIKI)
save_json(os.path.join(CORPUS_DIR, "medical_wiki.json"), medical_wiki * COMMON_CORPUS_REPEAT)
print(f"Medical-wiki corpus size: {len(medical_wiki)}")

alpaca_gpt4 = load_json(ALPACA_GPT4)
save_json(os.path.join(CORPUS_DIR, "alpaca_gpt4.json"), alpaca_gpt4 * COMMON_CORPUS_REPEAT)
print(f"Alpaca-GPT4 corpus size: {len(alpaca_gpt4)}")

pubmedqa = load_json(PUBMEDQA_TRAIN) + load_json(PUBMEDQA_TEST)
save_json(os.path.join(CORPUS_DIR, "PubMedQA.json"), pubmedqa * COMMON_CORPUS_REPEAT)
print(f"PubMedQA corpus size: {len(pubmedqa)}")

# Final merged corpus: oversample the domain corpora to balance the mix
final_merged = (cell_description_corpus
                + brain_region_corpus * BRAIN_REGION_REPEAT
                + cell_type_corpus * CELLTYPE_REPEAT)
print(f"Merged corpus size: {len(final_merged)}")

save_json(os.path.join(CORPUS_DIR, "STAB3_merged_v6.json"), final_merged)
print(f"Merged corpus saved to {os.path.join(CORPUS_DIR, 'STAB3_merged_v6.json')}")
print("Done.")
