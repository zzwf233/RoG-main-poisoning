#!/usr/bin/env bash
set -euo pipefail

# Quick check: run only CLEAN path to verify whether baseline can approach paper.
#
# Usage:
#   MODEL_PATH=/abs/path/to/RoG-model bash scripts/run_clean_alignment_check.sh
#   MODEL_PATH=/abs/path/to/RoG-model DATASETS=cwq bash scripts/run_clean_alignment_check.sh

DATASETS=${DATASETS:-"cwq,webqsp"}
MODEL_NAME=${MODEL_NAME:-RoG}
MODEL_PATH=${MODEL_PATH:-./RoG-model}
RULE_ROOT=${RULE_ROOT:-results/gen_rule_path}
PRED_ROOT=${PRED_ROOT:-results/KGQA}
CWQ_CLEAN=${CWQ_CLEAN:-datasets/clean_cwq.jsonl}
WEBQSP_CLEAN=${WEBQSP_CLEAN:-datasets/clean_webqsp.jsonl}

model_sanity() {
  local mp="$1"
  echo "[model] MODEL_PATH=${mp}"
  if [[ ! -d "$mp" ]]; then
    echo "[model][error] path is not a directory." >&2
    return 1
  fi

  local has_full=0
  [[ -f "${mp}/pytorch_model.bin" || -f "${mp}/model.safetensors" || -f "${mp}/pytorch_model.bin.index.json" || -f "${mp}/model.safetensors.index.json" ]] && has_full=1
  local has_adapter=0
  [[ -f "${mp}/adapter_config.json" || -f "${mp}/adapter_model.bin" || -f "${mp}/adapter_model.safetensors" ]] && has_adapter=1

  echo "[model] config.json: $([[ -f "${mp}/config.json" ]] && echo yes || echo no)"
  echo "[model] full weights: $([[ "$has_full" == "1" ]] && echo yes || echo no)"
  echo "[model] adapter files: $([[ "$has_adapter" == "1" ]] && echo yes || echo no)"

  if [[ "$has_adapter" == "1" && "$has_full" == "0" ]]; then
    echo "[model][warn] adapter-only checkpoint detected." >&2
    echo "             current clean check uses predict_answer.py with Llama loader (full-model path)." >&2
    echo "             if this is LoRA-only, scores can be far below paper unless merged/full weights are used." >&2
  fi
}

model_sanity "$MODEL_PATH"

run_dataset() {
  local d="$1"
  [[ ",${DATASETS}," == *",${d},"* ]]
}

dataset_clean_file() {
  local d="$1"
  if [[ "$d" == "cwq" ]]; then
    echo "${CWQ_CLEAN}"
  else
    echo "${WEBQSP_CLEAN}"
  fi
}

for d in cwq webqsp; do
  run_dataset "$d" || continue

  clean_file="$(dataset_clean_file "$d")"
  if [[ ! -f "$clean_file" ]]; then
    echo "[error] missing clean file for ${d}: $clean_file" >&2
    exit 2
  fi

  echo "=== [${d}] CLEAN alignment check ==="
  MODEL_PATH="$MODEL_PATH" DATASETS="$d" STAGES="rule,predict_clean" \
    CWQ_CLEAN="$CWQ_CLEAN" WEBQSP_CLEAN="$WEBQSP_CLEAN" \
    bash scripts/run_table3_paper_aligned.sh

  clean_tag="$(basename "$clean_file" .jsonl)"
  rule_file="${RULE_ROOT}/${clean_tag}/${MODEL_NAME}/test/predictions_3_False.jsonl"
  rule_postfix="$(echo "$rule_file" | tr '/.' '__')"
  eval_txt="${PRED_ROOT}/${d}-clean-paper/${MODEL_NAME}/test/${rule_postfix}/eval_result.txt"
  args_txt="${PRED_ROOT}/${d}-clean-paper/${MODEL_NAME}/test/${rule_postfix}/args.txt"

  echo "[${d}] args: ${args_txt}"
  [[ -f "$args_txt" ]] && cat "$args_txt"
  echo "[${d}] eval: ${eval_txt}"
  [[ -f "$eval_txt" ]] && cat "$eval_txt"
done
