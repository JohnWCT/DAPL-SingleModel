#!/usr/bin/env bash
# repro_MDSM：C_prototypical + D_summary 分類流程（100 / 500 epoch）
# 一律在 Docker container DAPL 內執行，不修改本機 Python 環境。
set -euo pipefail

CONTAINER="${CONTAINER:-DAPL}"
ROOT="${ROOT:-/workspace/DAPL-master}"

# ---------------------------------------------------------------------------
# 共用路徑（A/B 階段沿用既有 repro_MDSM 輸出，不重跑）
# ---------------------------------------------------------------------------
A_PRETRAIN="${ROOT}/output_dir/repro_MDSM_A_pretrain"
B_DRUG_LATENT="${ROOT}/output_dir/repro_MDSM_B_precontext/drug_latent_representation.pkl"

# Source GT：容器內無 ModelID966 檔時，改用 MaxScreen_raw（欄位對應見下方）
GT_MODELID966="${ROOT}/data/GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID966 drug230 samples201288.csv"
GT_MAXSCREEN_RAW="${ROOT}/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv"

if [[ -f "${GT_MODELID966}" ]]; then
  GT_INPUT="${GT_MODELID966}"
  GT_SAMPLE_COL="ModelID"
else
  GT_INPUT="${GT_MAXSCREEN_RAW}"
  GT_SAMPLE_COL="Sample_ID"
fi

# TCGA target eval（主 + 兩個次要）
TCGA_MAIN="${ROOT}/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv"
TCGA_ONLY="${ROOT}/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv"
TCGA_DAPL="${ROOT}/data/TCGA/TCGA_drug_response_from_DAPL.csv"

DRUG_SMILES="${ROOT}/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
CCLE_CANCER="${ROOT}/data_Winnie/CCLE_cancer_type.csv"
TCGA_CANCER="${ROOT}/data_Winnie/TCGA_cancer_type.csv"

# ---------------------------------------------------------------------------
# 刪除舊的 C/D 輸出（保留 A_pretrain / B_precontext）
# ---------------------------------------------------------------------------
delete_old_outputs() {
  local dirs=(
    "${ROOT}/output_dir/repro_MDSM_C_prototypical_classification_100epoch"
    "${ROOT}/output_dir/repro_MDSM_C_prototypical_classification_500epoch"
    "${ROOT}/output_dir/repro_MDSM_D_summary_classification_100epoch"
    "${ROOT}/output_dir/repro_MDSM_D_summary_classification_500epoch"
  )
  echo "[cleanup] 刪除舊輸出目錄 ..."
  docker exec "${CONTAINER}" bash -lc "
    set -e
    for d in ${dirs[*]}; do
      if [[ -d \"\$d\" ]]; then
        rm -rf \"\$d\"
        echo \"  removed: \$d\"
      else
        echo \"  skip (not found): \$d\"
      fi
    done
  "
}

# ---------------------------------------------------------------------------
# Step 1：C_prototypical
#   - 掃描 A_pretrain 下 12 組 latent，各組做 4 種 ftlr × scheduler grid
#   - 5-fold CV + CCLE independent test + 三份 TCGA target eval
#   - 結束後寫 best_model_summary.csv
# ---------------------------------------------------------------------------
run_c_prototypical() {
  local epochs="$1"
  local out_c="${ROOT}/output_dir/repro_MDSM_C_prototypical_classification_${epochs}epoch"

  echo "[C] epochs=${epochs} -> ${out_c}"
  echo "[C] gt_input=${GT_INPUT} (sample_col=${GT_SAMPLE_COL})"

  docker exec -w "${ROOT}" "${CONTAINER}" python3 C_prototypical.py \
    --task_type classification \
    --gt_input "${GT_INPUT}" \
    --prism_sample_id_col "${GT_SAMPLE_COL}" \
    --prism_drug_id_col drug_name \
    --binary_label_col Label \
    --tcga_eval_format tcga_dapl \
    --tcga_eval_gdsc_intersect13 "${TCGA_MAIN}" \
    --tcga_eval_tcga_only3 "${TCGA_ONLY}" \
    --tcga_eval_dapl "${TCGA_DAPL}" \
    --tcga_sample_id_col Patient_id \
    --tcga_drug_name_col drug_name \
    --tcga_label_col Label \
    --tcga_cancer_type_col cancers \
    --pretrain_dir "${A_PRETRAIN}" \
    --drug_latent_pkl "${B_DRUG_LATENT}" \
    --drug_smiles_input "${DRUG_SMILES}" \
    --drug_id_col drug_name \
    --drug_name_col DRUG_NAME \
    --random_seed 42 \
    --epochs "${epochs}" \
    --test_size 0.1 \
    --train_batch_size 1024 \
    --output_dir "${out_c}"
}

# ---------------------------------------------------------------------------
# Step 2：D_summary
#   - 讀 C 的 best_model_summary.csv，複製最佳組合
#   - 繪製 t-SNE，並重算 SSDA 風格 macro/weighted/overall 彙總表
# ---------------------------------------------------------------------------
run_d_summary() {
  local epochs="$1"
  local in_c="${ROOT}/output_dir/repro_MDSM_C_prototypical_classification_${epochs}epoch"
  local out_d="${ROOT}/output_dir/repro_MDSM_D_summary_classification_${epochs}epoch"

  echo "[D] c_output=${in_c}"
  echo "[D] output=${out_d}"

  docker exec -w "${ROOT}" "${CONTAINER}" python3 D_summary.py \
    --a_output_dir "${A_PRETRAIN}" \
    --c_output_dir "${in_c}" \
    --ccle_cancer_type_input "${CCLE_CANCER}" \
    --tcga_cancer_type_input "${TCGA_CANCER}" \
    --task_type classification \
    --write_ssda_eval 1 \
    --output_dir "${out_d}"
}

# ---------------------------------------------------------------------------
# Step 3（可選）：E_tcga_eval_by_drug — 主 target 各藥 per-drug mean/std
# ---------------------------------------------------------------------------
run_e_tcga_by_drug() {
  local epochs="$1"
  local out_d="${ROOT}/output_dir/repro_MDSM_D_summary_classification_${epochs}epoch"

  echo "[E] per-drug TCGA eval -> ${out_d}"
  docker exec -w "${ROOT}" "${CONTAINER}" python3 E_tcga_eval_by_drug.py \
    --d_summary_dir "${out_d}" \
    --predictions_rel_path copied_best_C_prototypical/target_eval_predictions.csv \
    --output_csv "${out_d}/target_eval_by_drug_fold_mean_std.csv"
}

# ---------------------------------------------------------------------------
# 單一 epoch 完整流程
# ---------------------------------------------------------------------------
run_one_epoch() {
  local epochs="$1"
  run_c_prototypical "${epochs}"
  run_d_summary "${epochs}"
  run_e_tcga_by_drug "${epochs}"
  echo "[done] repro_MDSM classification ${epochs}epoch pipeline finished."
}

usage() {
  cat <<EOF
用法:
  $0 cleanup                     # 只刪除四個舊輸出目錄
  $0 100                         # 重跑 100 epoch（C -> D -> E）
  $0 500                         # 重跑 500 epoch（C -> D -> E）
  $0 all                         # 先 cleanup，再依序跑 100 與 500

環境變數:
  CONTAINER  Docker 容器名稱（預設 DAPL）
  ROOT       容器內專案根目錄（預設 /workspace/DAPL-master）
EOF
}

main() {
  local cmd="${1:-all}"
  case "${cmd}" in
    cleanup)
      delete_old_outputs
      ;;
    100|500)
      run_one_epoch "${cmd}"
      ;;
    all)
      delete_old_outputs
      run_one_epoch 100
      run_one_epoch 500
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "未知指令: ${cmd}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
