#!/usr/bin/env bash
set -euo pipefail

# Fast diagnostic (no LLM):
# Evaluate rule usefulness by letting predict_answer output entities directly from paths.
# This isolates "rule quality" from "LLM answer generation quality".
#
# Usage:
#   DATASETS=cwq,webqsp bash scripts/probe_rule_ceiling.sh

DATASETS=${DATASETS:-"cwq,webqsp"}
MODEL_NAME=${MODEL_NAME:-RoG}
RULE_ROOT=${RULE_ROOT:-results/gen_rule_path}
PRED_ROOT=${PRED_ROOT:-results/KGQA}
CWQ_CLEAN=${CWQ_CLEAN:-datasets/clean_cwq.jsonl}
WEBQSP_CLEAN=${WEBQSP_CLEAN:-datasets/clean_webqsp.jsonl}

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
  clean_tag="$(basename "$clean_file" .jsonl)"
  rule_file="${RULE_ROOT}/${clean_tag}/${MODEL_NAME}/test/predictions_3_False.jsonl"

  if [[ ! -f "$clean_file" || ! -f "$rule_file" ]]; then
    echo "[error] missing clean/rule file for ${d}" >&2
    echo "        clean=${clean_file}" >&2
    echo "        rule=${rule_file}" >&2
    exit 2
  fi

  echo "=== [${d}] rule-ceiling probe (no-llm) ==="
  python src/qa_prediction/predict_answer.py \
    --data_path "$clean_file" \
    --d "${d}-rule-ceiling" \
    --split test \
    --model_name no-llm \
    --prompt_path prompts/llama2_predict.txt \
    --add_rule \
    --rule_path "$rule_file" \
    --predict_path "$PRED_ROOT" \
    --force
done

