#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-V3-0324}"
API_BASE="${API_BASE:-${OPENAI_BASE_URL:-https://api.siliconflow.cn/v1}}"
PROMPT_PATH="${PROMPT_PATH:-prompts/general_prompt.txt}"
PRED_ROOT="${PRED_ROOT:-results/KGQA_table5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
API_TIMEOUT="${API_TIMEOUT:-60}"
RETRY="${RETRY:-5}"
FORCE="${FORCE:-1}"

if [[ -z "${OPENAI_API_KEY:-${SILICONFLOW_API_KEY:-${DEEPSEEK_API_KEY:-}}}" ]]; then
  read -rsp "Enter API key for ${API_BASE}: " OPENAI_API_KEY
  echo
  export OPENAI_API_KEY
fi

force_args=()
if [[ "$FORCE" == "1" ]]; then
  force_args+=(--force)
fi

run_predict() {
  local data_path="$1"
  local dataset_name="$2"
  local rule_path="$3"

  python src/qa_prediction/predict_answer.py \
    --data_path "$data_path" \
    --d "$dataset_name" \
    --split test \
    --predict_path "$PRED_ROOT" \
    --model_name "$MODEL_NAME" \
    --prompt_path "$PROMPT_PATH" \
    --add_rule \
    --rule_path "$rule_path" \
    --api_base "$API_BASE" \
    --api_timeout "$API_TIMEOUT" \
    --retry "$RETRY" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${force_args[@]}"
}

echo "[1/4] DeepSeek clean prediction"
run_predict \
  "datasets/clean_webqsp.jsonl" \
  "table5-webqsp-clean" \
  "results/gen_rule_path/clean_webqsp/RoG/test/predictions_3_False.jsonl"

echo "[2/4] DeepSeek attack prediction"
run_predict \
  "datasets/poisoned_webqsp_ours.jsonl" \
  "table5-webqsp-attack" \
  "results/gen_rule_path/webqsp_full/RoG/test/predictions_3_False.jsonl"

rule_clean="results_gen_rule_path_clean_webqsp_RoG_test_predictions_3_False_jsonl"
rule_attack="results_gen_rule_path_webqsp_full_RoG_test_predictions_3_False_jsonl"
clean_pred="${PRED_ROOT}/table5-webqsp-clean/${MODEL_NAME}/test/${rule_clean}/predictions.jsonl"
attack_pred="${PRED_ROOT}/table5-webqsp-attack/${MODEL_NAME}/test/${rule_attack}/predictions.jsonl"

echo "[3/4] Evaluate clean"
python scripts/evaluate_table5_with_parser.py --prediction_file "$clean_pred"

echo "[4/4] Evaluate attack"
python scripts/evaluate_table5_with_parser.py --prediction_file "$attack_pred"

echo "Done."
echo "Clean predictions:  $clean_pred"
echo "Attack predictions: $attack_pred"
