#!/usr/bin/env python3
"""
=============================================================================
File      : 02_qc_merge.py
Directory : 02_qc
Purpose   : Assemble unified per-cell metadata (doublet calls, public/in-house annotations, brain-region hierarchy) and merge all samples into a single AnnData object.
=============================================================================
"""

import os
from multiprocessing import Pool

import pandas as pd
import scanpy as sc

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = "/path/to/your/project"  # <-- MODIFY THIS

data_dir = os.path.join(proj_dir, "data", "human", "qc")
out_dir = os.path.join(proj_dir, "data", "human", "output", "MergeData")

# List all sample directories
dir_list = []
for file_i in os.listdir(data_dir):
    dir_i = os.path.join(data_dir, file_i)
    if os.path.isdir(dir_i):
        dir_list.append(file_i)

dir_list.sort()

# Load per-sample doublet calls
doublets_list = []
for dir_i in dir_list:
    doublets_i = pd.read_csv(
        os.path.join(data_dir, dir_i, "doublets.csv"),
        header=None, sep="\t", index_col=0
    )
    doublets_i.columns = ["doublets_score", "doublets_prediction"]
    doublets_list.append(doublets_i)

meta_doublets = pd.concat(doublets_list, axis=0)


# Load public metadata
meta_public = pd.read_csv(
    os.path.join(data_dir, "public_meta.csv"),
    index_col=0
)

meta_public["species"] = "Human"
meta_public["diagnosis"] = "Control"
meta_public["sex"][meta_public["sex"] == "Unknown"] = "unknown"

relationship_counts = meta_public.groupby(['region', 'subregion']).size().reset_index(name='count')
relationship_counts.to_csv(
    os.path.join(data_dir, "public_region.csv"),
    sep=",", index=False
)

# Platform of each public dataset
mapping_platform = pd.DataFrame({
    "hs22": ["10X"],
    "hs30": ["10X"],
    "hs31": ["10X"],
    "hs32": ["10X"],
    "hs33": ["10X"],
    "hs34": ["10X"],
    "hs35": ["SPLiT-seq"],
    "hs36": ["10X"],
    "hs37": ["10X"],
    "hs38": ["STRT-seq"],
    "hs39": ["10X"],
    "hs40": ["Smart-seq"],
    "hs41": ["SPLiT-seq"],
    "hs43": ["Fluidigm C1"]
}, index=["platform"]).T

meta_public["dataset"] = meta_public["dataset"].astype(str)
meta_public["platform"] = mapping_platform.loc[meta_public["dataset"], "platform"].values

# Brain region hierarchy for public datasets:
# original subregion -> region_1 (coarse) / region_2 (fine) with abbreviations
mapping_region = pd.DataFrame({
    "Amygdala": ["Temporal lobe", "TL", "Amygdala", "AMG"],
    "Basolateral nuclear group (BLN) - lateral nucleus - La": ["Temporal lobe", "TL", "Lateral nucleus", "La"],
    "Bed nucleus of stria terminalis and nearby - BNST": ["Temporal lobe", "TL", "Bed nucleus of stria terminalis and nearby", "BNST"],
    "Central nuclear group - CEN": ["Temporal lobe", "TL", "Central nuclear group", "CEN"],
    "Corticomedial nuclear group (CMN) - anterior cortical nucleus - CoA": ["Temporal lobe", "TL", "Anterior cortical nucleus", "CoA"],
    "basolateral nuclear group (BLN) - basolateral nucleus (basal nucleus) - BL": ["Temporal lobe", "TL", "Basolateral nucleus", "BL"],
    "basolateral nuclear group (BLN) - basomedial nucleus (accessory basal nucleus) - BM": ["Temporal lobe", "TL", "Basomedial nucleus", "BM"],
    "corticomedial nuclear group - CMN": ["Temporal lobe", "TL", "Corticomedial nuclear group", "CMN"],
    "Basal ganglia": ["Basal ganglia", "BG", "Basal ganglia", "BG"],
    "Body of the Caudate - CaB": ["Basal ganglia", "BG", "Body of the Caudate", "CaB"],
    "Globus pallidus (GP) - External segment of globus pallidus - GPe": ["Basal ganglia", "BG", "External segment of globus pallidus", "Gpe"],
    "Globus pallidus (GP) - Internal segment of globus pallidus - GPi": ["Basal ganglia", "BG", "Internal segment of globus pallidus", "Gpi"],
    "Nucleus Accumbens - NAC": ["Basal ganglia", "BG", "Nucleus Accumbens", "NAC"],
    "Nucleus accumbens": ["Basal ganglia", "BG", "Nucleus Accumbens", "NAC"],
    "Putamen - Pu": ["Basal ganglia", "BG", "Putamen", "Pu"],
    "septal nuclei - SEP": ["Basal ganglia", "BG", "Septal nuclei", "SEP"],
    "substantia innominata and nearby nuclei - SI": ["Basal ganglia", "BG", "Substantia innominata and nearby nuclei", "SI"],
    "Brain": ["Whole Brain", "Whole Brain", "Whole Brain", "Whole Brain"],
    "Cerebellar Vermis - CBV": ["Cerebellum", "CB", "Cerebellar Vermis", "CBV"],
    "Cerebellar deep nuclei - CbDN": ["Cerebellum", "CB", "Cerebellar deep nuclei", "CbDN"],
    "Cerebellum": ["Cerebellum", "CB", "Cerebellum", "CB"],
    "Lateral hemisphere of cerebellum - CBL": ["Cerebellum", "CB", "Lateral hemisphere of cerebellum", "CBL"],
    "Unclassified": ["Cerebellum", "CB", "Cerebellum", "CB"],
    "Anterior Olfactory Nucleus - AON": ["Cerebral cortex", "CTX", "Anterior Olfactory Nucleus", "AON"],
    "Anterior cingulate cortex - ACC": ["Cerebral cortex", "CTX", "Anterior cingulate cortex", "ACC"],
    "Anterior parahippocampal gyrus (AG) - Lateral entorhinal cortex - LEC": ["Cerebral cortex", "CTX", "Lateral entorhinal cortex", "LEC"],
    "Anterior parahippocampal gyrus/posterior part (APH) - Medial entorhinal cortex - MEC": ["Cerebral cortex", "CTX", "Medial entorhinal cortex", "MEC"],
    "Caudal cingulate gyrus (CgGC) - A23": ["Cerebral cortex", "CTX", "Caudal cingulate gyrus", "A23"],
    "Central": ["Cerebral cortex", "CTX", "Cerebral cortex", "CTX"],
    "Cerebral cortex": ["Cerebral cortex", "CTX", "Cerebral cortex", "CTX"],
    "Cingulate gyrus/retrosplenial (CgGrs) - A29-A30": ["Cerebral cortex", "CTX", "Cingulate gyrus/retrosplenial", "A29-A30"],
    "Cuneus/caudal part - Peristriate Cortex - V2": ["Occipital lobe", "OL", "Peristriate Cortex", "V2"],
    "Cuneus/rostral part - Area Prostriata - Pro": ["Occipital lobe", "OL", "Area Prostriata", "Pro"],
    "Deeper lateral prefrontal cortex": ["Frontal lobe", "FL", "Deeper lateral prefrontal cortex", "DL-PFC"],
    "Deeper motor cortex": ["Frontal lobe", "FL", "Deeper motor cortex", "DMC"],
    "Deeper parietal lobe": ["Parietal lobe", "PL", "Deeper parietal cortex", "DPC"],
    "Deeper prefrontal cortex": ["Frontal lobe", "FL", "Deeper prefrontal cortex", "DPFC"],
    "Dorsalateral prefrontal cortex": ["Frontal lobe", "FL", "Dorsolateral prefrontal cortex", "DLPFC"],
    "Dorsolateral prefrontal cortex": ["Frontal lobe", "FL", "Dorsolateral prefrontal cortex", "DLPFC"],
    "Frontal agranular insular cortex - FI": ["Frontal lobe", "FL", "Frontal agranular insular cortex", "FI"],
    "Frontal cortex": ["Frontal lobe", "FL", "Frontal cortex", "FC"],
    "Gyrus rectus (ReG) - Medial orbitofrontal cortex - A14": ["Frontal lobe", "FL", "Medial orbitofrontal cortex", "A14"],
    "Hippocampus": ["Hippocampus", "HIP", "Hippocampus", "HIP"],
    "Inferior frontal gyrus (IFG) - Ventrolateral prefrontal cortex - A44-A45": ["Frontal lobe", "FL", "Ventrolateral prefrontal cortex", "A44-A45"],
    "Inferior temporal gyrus - ITG": ["Temporal lobe", "TL", "Inferior temporal cortex", "ITC"],
    "Lateral prefrontal cortex": ["Frontal lobe", "FL", "Lateral prefrontal cortex", "LPFC"],
    "Lingual gyrus (LiG) - Primary Visual Cortex - V1C": ["Occipital lobe", "OL", "Primary Visual Cortex", "V1C"],
    "Long insular gyri (LIG) - Dysgranular insular cortex - Idg": ["Insular lobe", "IL", "Dysgranular insular cortex", "Dins"],
    "Middle Temporal Gyrus - MTG": ["Temporal lobe", "TL", "Middle temporal cortex", "MTC"],
    "Middle frontal gyrus (MFG) - A46": ["Frontal lobe", "FL", "Middle frontal cortex", "MFC"],
    "Motor cortex": ["Frontal lobe", "FL", "Motor cortex", "M1"],
    "Neocortex": ["Cerebral cortex", "CTX", "Neocortex", "NCT"],
    "Occipital lobe": ["Occipital lobe", "OL", "Occipital cortex", "OC"],
    "Occipitotemporal (fusiform) gyrus/temporal part (FuGt) - Temporal area TF": ["Temporal lobe", "TL", "Fusiform gyrus", "FG"],
    "Pallium": ["Cerebrum", "CBR", "Pallium", "Pallium"],
    "Paretal lobe": ["Parietal lobe", "PL", "Parietal cortex", "PC"],
    "Parietal lobe": ["Parietal lobe", "PL", "Parietal cortex", "PC"],
    "Parietal operculum (PaO) - Gustatory cortex - A43": ["Parietal lobe", "PL", "Gustatory cortex", "A43"],
    "Perirhinal gyrus (PRG) - A35-A36": ["Cerebral cortex", "CTX", "Perirhinal gyrus", "A35-A36"],
    "Perirhinal gyrus (PRG) - \xa0A35-A36": ["Cerebral cortex", "CTX", "Perirhinal gyrus", "A35-A36"],
    "Piriform cortex - Pir": ["Cerebral cortex", "CTX", "Piriform cortex", "Pir"],
    "Postcentral gyrus (PoCG) - Primary somatosensory cortex - S1C": ["Parietal lobe", "PL", "Primary somatosensory cortex", "S1C"],
    "Posterior intermediate orbital gyrus (POrG) - Caudal division of OFCi - A13": ["Frontal lobe", "FL", "Caudal division of OFCi", "A13"],
    "Posterior parahippocampal gyrus (PPH) - TH-TL": ["Cerebral cortex", "CTX", "Posterior parahippocampal gyrus", "TH-TL"],
    "Precentral gyrus (PrCG) - Primary motor cortex - M1C": ["Frontal lobe", "FL", "Primary motor cortex", "M1C"],
    "Prefrontal cortex": ["Frontal lobe", "FL", "Prefrontal cortex", "PFC"],
    "Rostral gyrus (RoG) - Dorsal division of MFC - A32": ["Frontal lobe", "FL", "Dorsal division of MFC", "A32"],
    "Short insular gyri - Granular insular cortex - Ig": ["Insular lobe", "IL", "Granular insular cortex", "GI"],
    "Somatosensory cortex": ["Parietal lobe", "PL", "Somatosensory cortex", "S1"],
    "Subcallosal Gyrus (SCG) - Subgenual (subcallosal) division of MFC - A25": ["Frontal lobe", "FL", "Subgenual (subcallosal) division of MFC", "A25"],
    "Subgenual anterior cingulate cortex": ["Frontal lobe", "FL", "Subgenual anterior cingulate cortex", "sgACC"],
    "Superior Temporal Gyrus - STG": ["Temporal lobe", "TL", "Superior temporal cortex", "STC"],
    "Superior occipital gyrus (SOG) - Areas 19 and MT - A19": ["Occipital lobe", "OL", "Peristriate Cortex", "A19"],
    "Supramarginal gyrus (SMG) - A40": ["Parietal lobe", "PL", "Supramarginal gyrus", "A40"],
    "Supraparietal lobule (SPL) - Posterosuperior (dorsal) parietal cortex - A5-A7": ["Parietal lobe", "PL", "Posterosuperior (dorsal) parietal cortex", "A5-A7"],
    "Telencephalon": ["Telencephalon", "Tel", "Telencephalon", "Tel"],
    "Temporal lobe": ["Temporal lobe", "TL", "Temporal cortex", "TC"],
    "Temporal pole (TP) - Temporopolar area - A38": ["Temporal lobe", "TL", "Temporopolar area", "A38"],
    "Transverse temporal gyrus (TTG) - Primary auditory cortex - A1C": ["Temporal lobe", "TL", "Primary auditory cortex", "A1C"],
    "Visual cortex": ["Occipital lobe", "OL", "Visual cortex", "VC"],
    "rostral subdivision of area 35 - A35r": ["Cerebral cortex", "CTX", "Rostral subdivision of area 35", "A35r"],
    "Diencephalon": ["Diencephalon", "Dien", "Diencephalon", "Dien"],
    "Hypothalamus": ["Diencephalon", "Dien", "Hypothalamus", "DT"],
    "Thalamus": ["Thalamus", "Thal", "Thalamus", "Thal"],
    "ETH": ["Epithalamus", "EP", "Epithalamus", "EP"],
    "Forebrain": ["Prosencephalon", "Pros", "Prosencephalon", "Pros"],
    "Subcortex": ["Prosencephalon", "Pros", "Subcortex", "SC"],
    "Ganglionic eminences": ["Telencephalon", "Tel", "Ganglionic eminence", "GE"],
    "Cla": ["Cerebrum", "CBR", "Grey matter", "GM"],
    "Head": ["Whole Brain", "Whole Brain", "Whole Brain", "Whole Brain"],
    "Hindbrain": ["Rhombencephalon", "Romb", "Rhombencephalon", "Romb"],
    "CA1": ["Hippocampus", "HIP", "Cornu ammonis 1", "CA1"],
    "CA1-3": ["Hippocampus", "HIP", "Cornu ammonis 1-3", "CA1-3"],
    "CA2-3": ["Hippocampus", "HIP", "Cornu ammonis 2-3", "CA2-3"],
    "Caudal Hippocampus - CA1-CA3": ["Hippocampus", "HIP", "cCA1-3", "cCA1-3"],
    "Caudal Hippocampus - CA4-DGC": ["Hippocampus", "HIP", "cCA4-DGC", "cCA4-DGC"],
    "DG-CA4": ["Hippocampus", "HIP", "CA4-DGC", "CA4-DGC"],
    "Rostral CA1-2": ["Hippocampus", "HIP", "Rostral CA1-2", "rCA1-2"],
    "Rostral CA1-CA3": ["Hippocampus", "HIP", "Rostral CA1-3", "rCA1-3"],
    "Rostral CA3": ["Hippocampus", "HIP", "Rostral CA3", "rCA3"],
    "Rostral DG-CA4": ["Hippocampus", "HIP", "Rostral DG-CA4", "rDG-CA4"],
    "Subicular cortex - Sub": ["Hippocampus", "HIP", "Subicular cortex", "Sub"],
    "Uncal CA1-CA3": ["Hippocampus", "HIP", "Uncal CA1-3", "uCA1-3"],
    "Uncal DG-CA4": ["Hippocampus", "HIP", "Uncal DG-CA4", "uDG-CA4"],
    "mammillary region of HTH (HTHma)": ["Hypothalamus", "HT", "Mammillary region of HTH", "HTHma"],
    "mammillary region of HTH (HTHma) - mammillary nucleus - MN": ["Hypothalamus", "HT", "Mammillary nucleus", "MN"],
    "mammillary region of HTH (HTHma) - tuberal region of HTH - HTHtub": ["Hypothalamus", "HT", "Tuberal region of HTH", "HTHtub"],
    "preoptic region of HTH - HTHpo": ["Hypothalamus", "HT", "Preoptic region of HTH", "HTHpo"],
    "preoptic region of HTH - HTHpo (medial preoptic nucleus/MPN) - supraoptic region of HTH - HTHso (paraventricular nucleus/PV)": ["Hypothalamus", "HT", "Preoptic and supraoptic region of HTH", "HTHpo-HTHso"],
    "supraoptic region of HTH - HTHso": ["Hypothalamus", "HT", "Supraoptic region of HTH", "HTHso"],
    "supraoptic region of HTH - HTHso (anterior hypothalamic nucleus/AHN) - tuberal region of HTH - HTHtub (ventromedial and dorsomedial hypothalamic nucleic nuclei/VMH/DMH)": ["Hypothalamus", "HT", "Supraoptic and tuberal region of HTH", "HTHso-HTHtub"],
    "supraoptic region of HTH - HTHso (anterior hypothalamic nucleus/AHN) - tuberal region of HTH - HTHtub (ventromedial hypothalamic nucleus/VMH)": ["Hypothalamus", "HT", "Supraoptic and tuberal region of HTH", "HTHso-HTHtub"],
    "tuberal region of hypothalamus - HTHtub": ["Hypothalamus", "HT", "Tuberal region of hypothalamus", "HTHtub"],
    "Medulla": ["Hypothalamus", "HT", "Medulla", "MDL"],
    "afferent nuclei of cranial nerves in medulla oblongata - MoAN": ["Medulla", "MDL", "Afferent nuclei of cranial nerves in medulla oblongata", "MoAN"],
    "medullary reticular formation (MoRF) & efferent nuclei of cranial nerves in the medulla oblongata (MoEN) - MoRF-MoEN": ["Medulla", "MDL", "Medullary reticular formation & efferent nuclei of cranial nerves in the medulla oblongata", "MoRF-MoEN"],
    "precerebellar nuclei (PrCbN) - inferior olive - IO": ["Medulla", "MDL", "Inferior olive", "IO"],
    "sensory relay nuclei in medulla oblongata - MoSR": ["Medulla", "MDL", "Sensory relay nuclei in medulla oblongata", "MoSR"],
    "Inferior colliculus and nearby nuclei - IC": ["Mesencephalon", "Mes", "Inferior colliculus and nearby nuclei", "IC"],
    "Midbrain": ["Mesencephalon", "Mes", "Mesencephalon", "Mes"],
    "Midbrain dorsal": ["Mesencephalon", "Mes", "Dorsal mesencephalon", "dMes"],
    "Midbrain ventral": ["Mesencephalon", "Mes", "Ventral mesencephalon", "vMes"],
    "Periaqueductal gray and Dorsal raphe nucleus - PAG-DR": ["Mesencephalon", "Mes", "Periaqueductal gray and Dorsal raphe nucleus", "PAG-DR"],
    "Periaqueductal gray and nearby nuclei - PAG": ["Mesencephalon", "Mes", "Periaqueductal gray and nearby nuclei", "PAG"],
    "Pretectal region - PTR": ["Mesencephalon", "Mes", "Pretectal region", "PTR"],
    "Red Nucleus - RN": ["Mesencephalon", "Mes", "Red Nucleus", "RN"],
    "Substantia Nigra - SN": ["Mesencephalon", "Mes", "Substantia Nigra", "SN"],
    "Substantia Nigra and Red Nucleus - SN-RN": ["Mesencephalon", "Mes", "Substantia Nigra and Red Nucleus", "SN-RN"],
    "Substantia Nigra/Red Nucleus/and nearby nuclei - SN-RN": ["Mesencephalon", "Mes", "Substantia Nigra/Red Nucleus/and nearby nuclei", "SN-RN"],
    "Superior colliculus and nearby nuclei - SC": ["Mesencephalon", "Mes", "Superior colliculus and nearby nuclei", "SC"],
    "Unclassified": ["Pituitary gland", "PG", "Pituitary gland", "PG"],
    "Pons": ["Pons", "Pons", "Pons", "Pons"],
    "Pontine Nucleus - PN": ["Pons", "Pons", "Pontine Nucleus", "PN"],
    "Pontine reticular formation (PnRF) - parabrachial nuclei - PB": ["Pons", "Pons", "Parabrachial nuclei", "PB"],
    "Pontine reticular formation - PnRF": ["Pons", "Pons", "Pontine reticular formation", "PnRF"],
    "afferent nuclei of cranial nerves in pons - PnAN": ["Pons", "Pons", "Afferent nuclei of cranial nerves in pons", "PnAN"],
    "efferent nuclei of cranial nerves in pons - PnEN": ["Pons", "Pons", "Efferent nuclei of cranial nerves in pons", "PnEN"],
    "other nuclei in pontine tegmentum (XPnTg) - dorsal tegmental nucleus - DTg": ["Pons", "Pons", "dorsal tegmental nucleus", "DTg"],
    "SpC": ["Spinal cord", "SpC", "Spinal cord", "SpC"],
    "Cortex": ["Cerebral cortex", "CTX", "Cerebral cortex", "CTX"],
    "Hippocampus": ["Hippocampus", "HIP", "Hippocampus", "HIP"],
    "Striatum": ["Striatum", "STR", "Striatum", "STR"],
    "Subpallium": ["Telencephalon", "Tel", "Subpallium", "SPL"],
    "Telencephalon": ["Telencephalon", "Tel", "Telencephalon", "Tel"],
    "Anterior nuclear complex - ANC": ["Thalamus", "Thal", "Anterior nuclear complex", "ANC"],
    "Dorsal lateral geniculate nucleus": ["Thalamus", "Thal", "Dorsal lateral geniculate nucleus", "dLGN"],
    "Lateral nuclear complex (LNC) - ventral anterior nucleus of thalamus - VA": ["Thalamus", "Thal", "Ventral anterior nucleus of thalamus", "VA"],
    "Lateral nuclear complex of thalamus (LNC) - ventral group of lateral nucleus - VLN": ["Thalamus", "Thal", "Ventral group of lateral nucleus", "VLN"],
    "Lateral nuclear complex of thalamus (LNC) - ventral group of lateral nucleus - VLN (VPM and VPL)": ["Thalamus", "Thal", "Ventral group of lateral nucleus", "VLN"],
    "Subthalamic nucleus and nearby - STH": ["Thalamus", "Thal", "Subthalamic nucleus and nearby", "STH"],
    "Ventral group of lateral nucleus (VLN) - ventral anterior nucleus of thalamus - VA - ventral lateral nucleus of thalamus - VL": ["Thalamus", "Thal", "Ventral anterior nucleus and ventral lateral nucleus of thalamus", "VA-VL"],
    "intralaminar nuclear complex (ILN) - posterior group of intralaminar nuclei (PILN) - centromedian and parafasicular nuclei - CM and Pf": ["Thalamus", "Thal", "Centromedian and parafasicular nuclei", "CM-Pf"],
    "intralaminar nuclear complex (ILN) - posterior group of intralaminar nuclei (PILN) - centromedian nucleus of thalamus - CM": ["Thalamus", "Thal", "Centromedian nucleus of thalamus", "CM"],
    "lateral nuclear complex of thalamus (LNC) - Pulvinar of thalamus - Pul": ["Thalamus", "Thal", "Pulvinar of thalamus", "Pul"],
    "lateral nuclear complex of thalamus (LNC) - lateral posterior nucleus of thalamus + ventral posterior lateral nucleus - LP + VPL": ["Thalamus", "Thal", "Lateral posterior nucleus and ventral posterior lateral nucleus", "LP-VPL"],
    "lateral nuclear complex of thalamus (LNC) - lateral posterior nucleus of thalamus - LP": ["Thalamus", "Thal", "Lateral posterior nucleus of thalamus", "LP"],
    "lateral nuclear complex of thalamus (LNC) - ventral posterior lateral nucleus - VPL": ["Thalamus", "Thal", "Ventral posterior lateral nucleus", "VPL"],
    "medial nuclear complex of thalamus (MNC) - mediodorsal nucleus of thalamus + reuniens nucleus (medioventral nucleus) of thalamus - MD + Re": ["Thalamus", "Thal", "Mediodorsal nucleus and reuniens nucleus of thalamus", "MD-Re"],
    "medial nuclear complex of thalamus (MNC) - mediodorsal nucleus of thalamus - MD": ["Thalamus", "Thal", "Mediodorsal nucleus of thalamus", "MD"],
    "posterior nuclear complex of thalamus (PoN) - lateral geniculate nucleus (LG)": ["Thalamus", "Thal", "Lateral geniculate nucleus", "LG"],
    "posterior nuclear complex of thalamus (PoN) - medial geniculate nuclei (MG)": ["Thalamus", "Thal", "Medial geniculate nuclei", "MG"]

}, index=["region_1", "region_1_abbr", "region_2", "region_2_abbr"]).T

meta_public["region_original"] = meta_public["subregion"].copy()
meta_public["subregion"] = meta_public["subregion"].astype(str)
meta_public["region_1"] = mapping_region.loc[meta_public["subregion"], "region_1"].values
meta_public["region_1_abbr"] = mapping_region.loc[meta_public["subregion"], "region_1_abbr"].values
meta_public["region_2"] = mapping_region.loc[meta_public["subregion"], "region_2"].values
meta_public["region_2_abbr"] = mapping_region.loc[meta_public["subregion"], "region_2_abbr"].values

meta_public = meta_public.drop(
    columns=['region', 'subregion', "cell_type"]
)
meta_public = meta_public[["dataset", "species", "individual", "sample", "region_original", "region_1", "region_1_abbr", "region_2", "region_2_abbr", "age", "sex", "diagnosis", "platform"]]


# Load in-house (cell-level) metadata
meta_cell = pd.read_csv(
    os.path.join(data_dir, "stab2_meta.csv"),
    index_col=0
)

meta_cell["sex"][meta_cell["sex"] == "unknown         "] = "unknown"
meta_cell["diagnosis"][meta_cell["diagnosis"] == "control"] = "Control"

# Region name mapping for in-house datasets (coarse -> fine)
mapping_region_1 = pd.DataFrame({
    "ACC": ["Cingulate cortex", "Anterior cingulate cortex"],
    "anterior_midbrain": ["Mesencephalon", "Anterior mesencephalon"],
    "Brodmann area 9": ["Frontal lobe", "Dorsolateral prefrontal cortex"],
    "bank superior temporal": ["Temporal lobe", "Bank superior temporal gyrus"],
    "CA1": ["Hippocampus", "Cornu ammonis 1"],
    "CA24": ["Hippocampus", "Cornu ammonis 2-4"],
    "caudal-middle-frontal": ["Frontal lobe", "Caudal middle frontal gyrus"],
    "caudal_thalamus": ["Thalamus", "Caudal thalamus"],
    "center_midbrain": ["Mesencephalon", "Center mesencephalon"],
    "central": ["Cerebrum", "Cerebral cortex"],
    "cerebellum": ["Cerebellum", "Cerebellum"],
    "Cerebellum": ["Cerebellum", "Cerebellum"],
    "Cerebral cortex": ["Cerebrum", "Cerebral cortex"],
    "cerebral organoid": ["Organoid", "Organoid"],
    "Choroid_0_Choroid Plexus": ["Cerebrum", "Choroid plexus"],
    "Choroid_All_0": ["Cerebrum", "Choroid"],
    "Cortical_plate": ["Cerebrum", "Cerebral cortex"],
    "cortical plate": ["Cerebrum", "Cerebral cortex"],
    "control_white_matter": ["Cerebrum", "White matter"],
    "cortex": ["Cerebrum", "Cerebral cortex"],
    "Cortex_0_0": ["Cerebrum", "Cerebral cortex"],
    "Cortex_0_prefrontal cortex": ["Frontal lobe", "Prefrontal cortex"],
    "Cortex_0_primary visual cortex": ["Occipital lobe", "Primary visual cortex"],
    "Cortex_All_0": ["Cerebrum", "Cerebral cortex"],
    "Cortex_All_prefrontal cortex": ["Frontal lobe", "Prefrontal cortex"],
    "Cortex_All_primary motor cortex": ["Frontal lobe", "Primary motor cortex"],
    "Cortex_All_primary visual cortex": ["Occipital lobe", "Primary visual cortex"],
    "Cortex_cortical plate_0": ["Cerebrum", "Cerebral cortex"],
    "Cortex_cortical plate_prefrontal cortex": ["Frontal lobe", "Prefrontal cortex"],
    "Cortex_cortical plate_primary motor cortex": ["Frontal lobe", "Primary motor cortex"],
    "Cortex_cortical plate_primary visual cortex": ["Occipital lobe", "Primary visual cortex"],
    "Cortex_germinal zone_0": ["Cerebrum", "Cerebral cortex"],
    "Cortex_germinal zone_prefrontal cortex": ["Frontal lobe", "Prefrontal cortex"],
    "Cortex_subventricular zone_0": ["Cerebrum", "Cerebral cortex"],
    "Cortex_subventricular zone_prefrontal cortex": ["Frontal lobe", "Prefrontal cortex"],
    "Cortex_subventricular zone_primary visual cortex": ["Occipital lobe", "Primary visual cortex"],
    "Cortex_ventricular zone_0": ["Cerebrum", "Cerebral cortex"],
    "Cortex_ventricular zone_prefrontal cortex": ["Frontal lobe", "Prefrontal cortex"],
    "Cortex_ventricular zone_primary visual cortex": ["Occipital lobe", "Primary visual cortex"],
    "Ctrl-Ctx": ["Cerebrum", "Cerebral cortex"],
    "DFC": ["Frontal lobe", "Dorsolateral prefrontal cortex"],
    "DG": ["Hippocampus", "Dentate gyrus"],
    "Diencephalon": ["Prosencephalon", "Diencephalon"],
    "dlPFC": ["Frontal lobe", "Dorsolateral prefrontal cortex"],
    "dorsolateral prefrontal cortex": ["Frontal lobe", "Dorsolateral prefrontal cortex"],
    "dorsal_thalamus": ["Thalamus", "Dorsal Thalamus"],
    "EC": ["Temporal lobe", "Entorhinal cortex"],
    "forebrain": ["Prosencephalon", "Prosencephalon"],
    "Frontal Cortex": ["Frontal lobe", "Frontal cortex"],
    "Frontal cortex": ["Frontal lobe", "Frontal cortex"],
    "frontal lobe": ["Frontal lobe", "Frontal cortex"],
    "frontal_lobe_of_CC": ["Frontal lobe", "Frontal cortex"],
    "fusiform": ["Temporal & Occipital lobe", "Fusiform gyrus"],
    "Ganglionic eminence": ["Prosencephalon", "Ganglionic eminence"],
    "germinal matrix": ["Cerebrum", "Cerebral cortex"],
    "GE": ["Prosencephalon", "Ganglionic eminence"],
    "GE_0_lateral ganglionic eminence": ["Prosencephalon", "Lateral ganglionic eminence"],
    "GE_0_medial ganglionic eminence": ["Prosencephalon", "Medial ganglionic eminence"],
    "GE_All_0": ["Prosencephalon", "ganglionic eminence"],
    "GE_All_lateral ganglionic eminence": ["Prosencephalon", "Lateral ganglionic eminence"],
    "GE_All_medial ganglionic eminence": ["Prosencephalon", "Medial ganglionic eminence"],
    "hindbrain": ["Rhombencephalon", "Rhombencephalon"],
    "Hindbrain": ["Rhombencephalon", "Rhombencephalon"],
    "Hippocampus": ["Hippocampus", "Hippocampus"],
    "hippocampus": ["Hippocampus", "Hippocampus"],
    "hypothalamus": ["Hypothalamus", "Hypothalamus"],
    "Hypothalamus": ["Hypothalamus", "Hypothalamus"],
    "inferior parietal": ["Parietal lobe", "Inferior parietal cortex"],
    "inferior temporal": ["Temporal lobe", "Inferior temporal cortex"],
    "insular gyrus": ["Insular lobe", "Insular cortex"],
    "lateral cerebellum": ["Cerebellum", "Lateral cerebellum"],
    "lateral occipital": ["Occipital lobe", "Lateral occipital cortex"],
    "left_hemisphere": ["Cerebrum", "Left brain"],
    "medulla": ["Brainstem", "Medulla"],
    "midbrain": ["Mesencephalon", "Mesencephalon"],
    "Midbrain": ["Mesencephalon", "Mesencephalon"],
    "middle temporal": ["Temporal lobe", "Middle temporal cortex"],
    "M1": ["Frontal lobe", "Primary motor cortex"],
    "NCX": ["Cerebrum", "Neocortex"],
    "neocortex": ["Cerebrum", "Neocortex"],
    "occipital": ["Occipital lobe", "Occipital cortex"],
    "Occipital": ["Occipital lobe", "Occipital cortex"],
    "Occipital Lobe, Visual Cortex": ["Occipital lobe", "Visual cortex"],
    "occipital_lobe_of_CC": ["Occipital lobe", "Occipital cortex"],
    "optic_cup": ["Retina", "Optic cup"],
    "optic_vesicle": ["Retina", "Optic vesicle"],
    "OpticVesicle": ["Retina", "Optic vesicle"],
    "pallium": ["Cerebrum", "Pallium"],
    "parietal_lobe_of_CC": ["Parietal lobe", "Parietal cortex"],
    "pars opercularis": ["Frontal lobe", "Pars opercularis"],
    "pars orbitalis": ["Frontal lobe", "Pars orbitalis"],
    "pars triangulars": ["Frontal lobe", "Pars triangularis"],
    "PFC": ["Frontal lobe", "Prefrontal cortex"],
    "pons": ["Brainstem", "Pons"],
    "PONS": ["Brainstem", "Pons"],
    "post-central": ["Parietal lobe", "Primary somatosensory cortex"],
    "pre-central": ["Frontal lobe", "Primary somatomotor cortex"],
    "precuneus": ["Parietal lobe", "Precuneus"],
    "Prefrontal cortex": ["Frontal lobe", "Prefrontal cortex"],
    "prefrontal_cortex_BA10": ["Frontal lobe", "Anterior prefrontal cortex"],
    "prefrontal_cortex_BA46": ["Frontal lobe", "Dorsolateral prefrontal cortex"],
    "prefrontal_cortex_BA8": ["Frontal lobe", "Frontal eye fields"],
    "prefrontal_cortex_BA9,46": ["Frontal lobe", "Dorsolateral prefrontal cortex"],
    "prefrontal_cortex_BA9": ["Frontal lobe", "Dorsolateral prefrontal cortex"],
    "Presumptive_cortex_0_0": ["Prosencephalon", "Presumptive cortex"],
    "Presumptive_cortex_All_0": ["Prosencephalon", "Presumptive cortex"],
    "prosencephalon": ["Prosencephalon", "Prosencephalon"],
    "pulvinar_thalamus": ["Thalamus", "Pulvinar thalamus"],
    "retina": ["Retina", "Retina"],
    "right_hemisphere": ["Cerebrum", "Right brain"],
    "rostral-middle-frontal ": ["Frontal lobe", "Rostral middle frontal gyrus"],
    "striatum": ["Striatum", "Striatum"],
    "SUB": ["Hippocampus", "Subiculum"],
    "subpallium": ["Cerebrum", "Subpallium"],
    "superior frontal": ["Frontal lobe", "Superior frontal cortex"],
    "superior parietal": ["Parietal lobe", "Superior parietal cortex"],
    "superior temporal": ["Temporal lobe", "Superior temporal cortex"],
    "supra-maginal": ["Parietal lobe", "Supramarginal gyrus"],
    "svz+caudate": ["Cerebrum", "Cerebral cortex"],
    "Telencephalon": ["Prosencephalon", "Telencephalon"],
    "temporal_lobe_of_CC": ["Temporal lobe", "Temporal cortex"],
    "thalamus": ["Thalamus", "Thalamus"],
    "Thalamus": ["Thalamus", "Thalamus"],
    "ventral_thalamus": ["Thalamus", "Ventral thalamus"],
    "vermis cerebellum": ["Cerebellum", "Vermis cerebellum"],
    "White Matter brain tissue": ["Cerebrum", "White matter"],
}, index=["coarse", "fine"]).T

meta_cell["region_original"] = meta_cell["region"].copy()
meta_cell["region"] = meta_cell["region"].astype(str)
meta_cell["region_fine"] = mapping_region_1.loc[meta_cell["region"], "fine"].values

# Region hierarchy for in-house datasets (fine -> region_1/region_2 with abbreviations)
mapping_region_2 = pd.DataFrame({
    "Cerebellum": ["Cerebellum", "CB", "Cerebellum", "CB"],
    "Lateral cerebellum": ["Cerebellum", "CB", "Lateral cerebellum", "LCB"],
    "Vermis cerebellum": ["Cerebellum", "CB", "Vermis cerebellum", "VCB"],
    "Anterior cingulate cortex": ["Cerebral cortex", "CTX", "Anterior cingulate cortex", "ACC"],
    "Cerebral cortex": ["Cerebral cortex", "CTX", "Cerebral cortex", "CTX"],
    "Neocortex": ["Cerebral cortex", "CTX", "Neocortex", "NCT"],
    "Choroid": ["Cerebrum", "CBR", "Choroid", "CR"],
    "Choroid plexus": ["Cerebrum", "CBR", "Choroid plexus", "CRP"],
    "Left brain": ["Cerebrum", "CBR", "Left brain", "LB"],
    "Pallium": ["Cerebrum", "CBR", "Pallium", "Pallium"],
    "Right brain": ["Cerebrum", "CBR", "Right brain", "RB"],
    "White matter": ["Cerebrum", "CBR", "White matter", "WM"],
    "Diencephalon": ["Diencephalon", "Dien", "Diencephalon", "Dien"],
    "Anterior prefrontal cortex": ["Frontal lobe", "FL", "Anterior prefrontal cortex", "APC"],
    "Caudal middle frontal gyrus": ["Frontal lobe", "FL", "Caudal middle frontal gyrus", "CMFG"],
    "Dorsolateral prefrontal cortex": ["Frontal lobe", "FL", "Dorsolateral prefrontal cortex", "DlPFC"],
    "Frontal cortex": ["Frontal lobe", "FL", "Frontal cortex", "FC"],
    "Frontal eye fields": ["Frontal lobe", "FL", "Frontal eye fields", "FEF"],
    "Pars opercularis": ["Frontal lobe", "FL", "Pars opercularis", "OP"],
    "Pars orbitalis": ["Frontal lobe", "FL", "Pars orbitalis", "ORB"],
    "Pars triangularis": ["Frontal lobe", "FL", "Pars triangularis", "TRI"],
    "Prefrontal cortex": ["Frontal lobe", "FL", "Prefrontal cortex", "PFC"],
    "Primary motor cortex": ["Frontal lobe", "FL", "Primary motor cortex", "M1C"],
    "Primary somatomotor cortex": ["Frontal lobe", "FL", "Primary somatomotor cortex", "PSC"],
    "Rostral middle frontal gyrus": ["Frontal lobe", "FL", "Rostral middle frontal gyrus", "RMFG"],
    "Superior frontal cortex": ["Frontal lobe", "FL", "Superior frontal cortex", "SFC"],
    "Cornu ammonis 1": ["Hippocampus", "HIP", "Cornu ammonis 1", "CA1"],
    "Cornu ammonis 2-4": ["Hippocampus", "HIP", "Cornu ammonis 2-4", "CA2-4"],
    "Dentate gyrus": ["Hippocampus", "HIP", "Dentate gyrus", "DG"],
    "Hippocampus": ["Hippocampus", "HIP", "Hippocampus", "HIP"],
    "Subiculum": ["Hippocampus", "HIP", "Subiculum", "Sub"],
    "Hypothalamus": ["Hypothalamus", "HT", "Hypothalamus", "HT"],
    "Insular cortex": ["Insular lobe", "IL", "Insular cortex", "IC"],
    "Medulla": ["Medulla", "MDL", "Medulla", "MDL"],
    "Anterior mesencephalon": ["Mesencephalon", "Mes", "Anterior mesencephalon", "aMes"],
    "Center mesencephalon": ["Mesencephalon", "Mes", "Center mesencephalon", "cMes"],
    "Mesencephalon": ["Mesencephalon", "Mes", "Mesencephalon", "Mes"],
    "Lateral occipital cortex": ["Occipital lobe", "OL", "Lateral occipital cortex", "LOC"],
    "Occipital cortex": ["Occipital lobe", "OL", "Occipital cortex", "OC"],
    "Primary visual cortex": ["Occipital lobe", "OL", "Primary visual cortex", "V1C"],
    "Visual cortex": ["Occipital lobe", "OL", "Visual cortex", "VC"],
    "Organoid": ["Organoid", "Organoid", "Organoid", "Organoid"],
    "Inferior parietal cortex": ["Parietal lobe", "PL", "Inferior parietal cortex", "IPC"],
    "Parietal cortex": ["Parietal lobe", "PL", "Parietal cortex", "PC"],
    "Precuneus": ["Parietal lobe", "PL", "Precuneus", "Precuneus"],
    "Primary somatosensory cortex": ["Parietal lobe", "PL", "Primary somatosensory cortex", "S1C"],
    "Superior parietal cortex": ["Parietal lobe", "PL", "Superior parietal cortex", "SPC"],
    "Supramarginal gyrus": ["Parietal lobe", "PL", "Supramarginal gyrus", "SMG"],
    "Pons": ["Pons", "Pons", "Pons", "Pons"],
    "Prosencephalon": ["Prosencephalon", "Pros", "Prosencephalon", "Pros"],
    "Optic cup": ["Retina", "Retina", "Optic cup", "OPC"],
    "Optic vesicle": ["Retina", "Retina", "Optic vesicle", "OPV"],
    "Retina": ["Retina", "Retina", "Retina", "Retina"],
    "Rhombencephalon": ["Rhombencephalon", "Romb", "Rhombencephalon", "RCL"],
    "Striatum": ["Striatum", "STR", "Striatum", "STR"],
    "Ganglionic eminence": ["Telencephalon", "Tel", "Ganglionic eminence", "GE"],
    "ganglionic eminence": ["Telencephalon", "Tel", "Ganglionic eminence", "GE"],
    "Lateral ganglionic eminence": ["Telencephalon", "Tel", "Lateral ganglionic eminence", "LGE"],
    "Medial ganglionic eminence": ["Telencephalon", "Tel", "Medial ganglionic eminence", "MGE"],
    "Presumptive cortex": ["Telencephalon", "Tel", "Presumptive cortex", "PRC"],
    "Subpallium": ["Telencephalon", "Tel", "Subpallium", "SPL"],
    "Telencephalon": ["Telencephalon", "Tel", "Telencephalon", "Tel"],
    "Fusiform gyrus": ["Temporal & Occipital lobe", "TOL", "Fusiform gyrus", "FG"],
    "Bank superior temporal gyrus": ["Temporal lobe", "TL", "Bank superior temporal gyrus", "BSTG"],
    "Entorhinal cortex": ["Temporal lobe", "TL", "Entorhinal cortex", "EC"],
    "Inferior temporal cortex": ["Temporal lobe", "TL", "Inferior temporal cortex", "ITC"],
    "Middle temporal cortex": ["Temporal lobe", "TL", "Middle temporal cortex", "MTC"],
    "Superior temporal cortex": ["Temporal lobe", "TL", "Superior temporal cortex", "STC"],
    "Temporal cortex": ["Temporal lobe", "TL", "Temporal cortex", "TC"],
    "Caudal thalamus": ["Thalamus", "Thal", "Caudal thalamus", "cThal"],
    "Dorsal Thalamus": ["Thalamus", "Thal", "Dorsal Thalamus", "dThal"],
    "Pulvinar thalamus": ["Thalamus", "Thal", "Pulvinar thalamus", "pThal"],
    "Thalamus": ["Thalamus", "Thal", "Thalamus", "Thal"],
    "Ventral thalamus": ["Thalamus", "Thal", "Ventral thalamus", "vThal"]
}, index=["region_1", "region_1_abbr", "region_2", "region_2_abbr"]).T

meta_cell["region_fine"] = meta_cell["region_fine"].astype(str)
meta_cell["region_1"] = mapping_region_2.loc[meta_cell["region_fine"], "region_1"].values
meta_cell["region_1_abbr"] = mapping_region_2.loc[meta_cell["region_fine"], "region_1_abbr"].values
meta_cell["region_2"] = mapping_region_2.loc[meta_cell["region_fine"], "region_2"].values
meta_cell["region_2_abbr"] = mapping_region_2.loc[meta_cell["region_fine"], "region_2_abbr"].values

meta_cell = meta_cell.drop(
    columns=["orig.ident", "nCount_RNA", "nFeature_RNA", "region", "region_fine"]
)
meta_cell = meta_cell[["dataset", "species", "individual", "sample", "region_original", "region_1", "region_1_abbr", "region_2", "region_2_abbr", "age", "sex", "diagnosis", "platform"]]

meta_all = pd.concat([meta_cell, meta_public], ignore_index=False)

meta = pd.concat([meta_all, meta_doublets], axis=1)
meta['doublets_prediction'] = meta['doublets_prediction'].fillna(1.0)

# Harmonize non-standard age values
age_corrections = {
    "50": "Y50", "60": "Y60", "63": "Y63", "65": "Y65", "69": "Y69", "70": "Y70", "72": "Y72", "81": "Y81", "M12": "Y1",
    "Y79_pmi4.25": "Y79", "Y79_pmi5": "Y79", "Y79_pmi7": "Y79", "Y79_pmiNA": "Y79", "Y83_pmi1.8": "Y83", "Y90_pmi2.92": "Y90", "Y90_pmi4": "Y90"
}
meta["age"] = meta["age"].astype(str)
meta["age"] = meta['age'].replace(age_corrections)

# Map age to developmental stage
stage_info = {
    "PCW4": "01-Embryonic",
    "PCW5": "01-Embryonic",
    "PCW5.85": "01-Embryonic",
    "PCW6": "01-Embryonic",
    "PCW7": "01-Embryonic",
    "PCW8": "02-Early fetal",
    "PCW9": "02-Early fetal",
    "PCW10": "03-Early fetal",
    "PCW11": "03-Early fetal",
    "PCW11.5": "03-Early fetal",
    "PCW12": "03-Early fetal",
    "PCW13": "04-Early midfetal",
    "PCW13.3": "04-Early midfetal",
    "PCW14": "04-Early midfetal",
    "PCW14.2": "04-Early midfetal",
    "PCW14.5": "04-Early midfetal",
    "PCW14.7": "04-Early midfetal",
    "PCW15": "04-Early midfetal",
    "PCW16": "05-Early midfetal",
    "PCW16.3": "05-Early midfetal",
    "PCW16.5": "05-Early midfetal",
    "PCW17": "05-Early midfetal",
    "PCW17.3": "05-Early midfetal",
    "PCW17.5": "05-Early midfetal",
    "PCW17.7": "05-Early midfetal",
    "PCW18": "05-Early midfetal",
    "PCW18.5": "05-Early midfetal",
    "PCW19": "06-Late midfetal",
    "PCW20": "06-Late midfetal",
    "PCW21": "06-Late midfetal",
    "PCW21.5": "06-Late midfetal",
    "PCW22": "06-Late midfetal",
    "PCW23": "06-Late midfetal",
    "PCW24": "07-Late fetal",
    "PCW25": "07-Late fetal",
    "PCW26": "07-Late fetal",
    "PCW27": "07-Late fetal",
    "PCW28": "07-Late fetal",
    "PCW32": "07-Late fetal",
    "PCW33": "07-Late fetal",
    "PCW34": "07-Late fetal",
    "PCW37": "07-Late fetal",
    "PCW38": "07-Late fetal",
    "D118": "08-Neonatal and early infancy",
    "D179": "08-Neonatal and early infancy",
    "D2": "08-Neonatal and early infancy",
    "D34": "08-Neonatal and early infancy",
    "D86": "08-Neonatal and early infancy",
    "M5": "08-Neonatal and early infancy",
    "Y0": "08-Neonatal and early infancy",
    "Y0.1": "08-Neonatal and early infancy",
    "Y0.2": "08-Neonatal and early infancy",
    "D301": "09-Late infancy",
    "Y0.6": "09-Late infancy",
    "M9": "09-Late infancy",
    "Y1": "10-Early childhood",
    "Y1.2": "10-Early childhood",
    "Y1.4": "10-Early childhood",
    "Y1.5": "10-Early childhood",
    "Y1.7": "10-Early childhood",
    "Y1.8": "10-Early childhood",
    "Y2": "10-Early childhood",
    "Y2.1": "10-Early childhood",
    "Y2.5": "10-Early childhood",
    "Y2.6": "10-Early childhood",
    "Y2.7": "10-Early childhood",
    "Y2.8": "10-Early childhood",
    "Y3": "10-Early childhood",
    "Y3.2": "10-Early childhood",
    "Y4": "10-Early childhood",
    "Y4.3": "10-Early childhood",
    "Y5.7": "10-Early childhood",
    "Y6": "11-Middle and late childhood",
    "Y6.7": "11-Middle and late childhood",
    "Y8": "11-Middle and late childhood",
    "Y10": "11-Middle and late childhood",
    "Y12": "12-Adolescence",
    "Y13": "12-Adolescence",
    "Y14": "12-Adolescence",
    "Y14.7": "12-Adolescence",
    "Y15": "12-Adolescence",
    "Y15.3": "12-Adolescence",
    "Y16": "12-Adolescence",
    "Y17": "12-Adolescence",
    "Y18": "12-Adolescence",
    "Y18.5": "12-Adolescence",
    "Y19": "12-Adolescence",
    "Y20": "13-Young adulthood",
    "Y21": "13-Young adulthood",
    "Y22": "13-Young adulthood",
    "Y25": "13-Young adulthood",
    "Y27": "13-Young adulthood",
    "Y28": "13-Young adulthood",
    "Y29": "13-Young adulthood",
    "Y32": "13-Young adulthood",
    "Y35": "13-Young adulthood",
    "Y36": "13-Young adulthood",
    "Y39": "13-Young adulthood",
    "Y40": "14-Middle adulthood",
    "Y42": "14-Middle adulthood",
    "Y43": "14-Middle adulthood",
    "Y44": "14-Middle adulthood",
    "Y45": "14-Middle adulthood",
    "Y47": "14-Middle adulthood",
    "Y48": "14-Middle adulthood",
    "Y49": "14-Middle adulthood",
    "Y50": "14-Middle adulthood",
    "Y50.2": "14-Middle adulthood",
    "Y51": "14-Middle adulthood",
    "Y52": "14-Middle adulthood",
    "Y53": "14-Middle adulthood",
    "Y54": "14-Middle adulthood",
    "Y55": "14-Middle adulthood",
    "Y56": "14-Middle adulthood",
    "Y59": "14-Middle adulthood",
    "Y60": "15-Late adulthood",
    "Y62": "15-Late adulthood",
    "Y63": "15-Late adulthood",
    "Y64": "15-Late adulthood",
    "Y65": "15-Late adulthood",
    "Y66": "15-Late adulthood",
    "Y69": "15-Late adulthood",
    "Y70": "15-Late adulthood",
    "Y71": "15-Late adulthood",
    "Y72": "15-Late adulthood",
    "Y73": "15-Late adulthood",
    "Y74": "15-Late adulthood",
    "Y75": "15-Late adulthood",
    "Y77": "15-Late adulthood",
    "Y78": "15-Late adulthood",
    "Y79": "15-Late adulthood",
    "Y80": "15-Late adulthood",
    "Y81": "15-Late adulthood",
    "Y82": "15-Late adulthood",
    "Y83": "15-Late adulthood",
    "Y84": "15-Late adulthood",
    "Y85": "15-Late adulthood",
    "Y86": "15-Late adulthood",
    "Y87": "15-Late adulthood",
    "Y88": "15-Late adulthood",
    "Y89": "15-Late adulthood",
    "Y90": "15-Late adulthood",
    "Y92": "15-Late adulthood",
    "Y93": "15-Late adulthood",
    "Y95": "15-Late adulthood"
}

meta['stage'] = meta['age'].map(stage_info)

meta.to_csv(
    os.path.join(data_dir, "meta_all.csv"),
    sep=",", index=True
)

# Load all 10X-format samples in parallel
def load_10x(path):
    adata = sc.read_10x_mtx(
        os.path.join(data_dir, path)
    )
    print(path + " finished!")
    return adata


pool = Pool(processes=32)
obj_list = [
    pool.apply_async(
        load_10x, args=(dir_i,)
    ) for dir_i in dir_list
]
pool.close()
pool.join()

obj_list = [
    obj_i.get() for obj_i in obj_list
]
ad_hs = sc.AnnData.concatenate(
    *obj_list, join="outer", index_unique=None
)

# Attach ensemble IDs from the reference gene list
gtf = pd.read_csv(
    os.path.join(proj_dir, "data", "ref", "human_reference_unique.txt"),
    sep="\t", header=None
)
gtf.columns = ["id", "name"]
gtf.index = gtf.name
ad_hs.var = ad_hs.var.drop(columns=ad_hs.var.columns)
ad_hs.var["ensemble_id"] = gtf.loc[ad_hs.var_names, "id"].values

# Transfer the unified metadata into the AnnData object
columns_to_convert = [
    "species", "dataset", "individual", "sample", "age", "stage", "sex",
    "region_original", "region_1", "region_1_abbr", "region_2", "region_2_abbr",
    "platform", "diagnosis"
]

for col in columns_to_convert:
    ad_hs.obs[col] = meta[col].astype(str)

ad_hs.obs["doublets_score"] = meta["doublets_score"].astype("float32")
ad_hs.obs["doublets_prediction"] = meta["doublets_prediction"].astype("bool")
ad_hs.obs = ad_hs.obs.drop(columns=["batch"])

# Convert any categorical columns in ad_hs.var to strings
for col in ad_hs.var.select_dtypes(['category']).columns:
    ad_hs.var[col] = ad_hs.var[col].astype(str)

# Save the merged data
ad_hs.write_h5ad(
    filename=os.path.join(out_dir, "hs_merged_all.h5ad")
)
ad_hs.obs.to_csv(
    os.path.join(out_dir, "hs_meta_merged.csv")
)
print("Saved merged data to", os.path.join(out_dir, "hs_merged_all.h5ad"))
