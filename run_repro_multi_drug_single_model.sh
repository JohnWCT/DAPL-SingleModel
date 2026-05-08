#!/usr/bin/env bash
set -euo pipefail

# Run DAPL pretraining steps inside Docker container `DAPL`.
# Unified output prefix keeps folders grouped and sortable.

CONTAINER_NAME="DAPL"
WORKSPACE_ROOT="/workspace/DAPL_git"
OUTPUT_PREFIX="repro_multi_drug_single_model"

echo "[1/2] Run A_pretrain (CCLE + TCGA)"
docker exec "${CONTAINER_NAME}" python3 "${WORKSPACE_ROOT}/A_pretrain.py" \
  --ccle_exp_input "${WORKSPACE_ROOT}/data/pretrain_ccle.csv" \
  --tcga_exp_input "${WORKSPACE_ROOT}/data/TCGA/pretrain_tcga.csv" \
  --output_dir "${WORKSPACE_ROOT}/output_dir/${OUTPUT_PREFIX}_A_pretrain" \
  --random_seed 42

echo "[2/2] Run B_precontext (drug graph pretraining)"
docker exec "${CONTAINER_NAME}" python3 "${WORKSPACE_ROOT}/B_precontext.py" \
  --drug_input "${WORKSPACE_ROOT}/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv" \
  --drug_id_col drug_name \
  --smiles_col SMILES \
  --drug_name_col DRUG_NAME \
  --output_dir "${WORKSPACE_ROOT}/output_dir/${OUTPUT_PREFIX}_B_precontext" \
  --epochs 100 \
  --batch_size 128 \
  --random_seed 42

echo "All jobs completed."
