#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# Paper-aligned workflow for sub-question cascading poisoning.
# This script preserves the existing pipeline and writes to separate outputs
# so you can compare against the current stronger setting safely.
#
# Alignment target from RAG Safety:
# - N = 5 adversarial answers per question
# - K = 4 perturbation triples per adversarial answer
# - total cap = N * K = 20 triples per question
#
# In our current implementation, the closest comparable controls are:
# - --num_candidates 5
# - --inject_top_k 5
# - --budget_k 20           (total triple cap per question)
# - repeat / boost terms disabled to avoid stronger-than-paper injection
#
# Staged usage examples:
#   STAGES=poison,predict,eval ORIGINAL_DATASET_PATH="datasets/clean_cwq.jsonl" bash scripts/subquestion-cascade-poison-ragsafety.sh
#   STAGES=rule bash scripts/subquestion-cascade-poison-ragsafety.sh
#   STAGES=predict,eval bash scripts/subquestion-cascade-poison-ragsafety.sh

STAGES=${STAGES:-"decompose,rule,poison,predict,eval"}

ORIGINAL_DATASET_PATH=${ORIGINAL_DATASET_PATH:-"datasets/clean_questions.jsonl"}
RUN_DECOMPOSE=${RUN_DECOMPOSE:-"1"}
DATASET_PATH=${DATASET_PATH:-"datasets/clean_subquestions_ragsafety.jsonl"}

RULE_OUTPUT_ROOT=${RULE_OUTPUT_ROOT:-"results/gen_rule_path"}
RULE_FILE=${RULE_FILE:-"${RULE_OUTPUT_ROOT}/clean_subquestions_ragsafety/RoG/test/predictions_3_False.jsonl"}
POISONED_DATASET_PATH=${POISONED_DATASET_PATH:-"datasets/poisoned_subquestions_ragsafety.jsonl"}
MODEL_PATH=${MODEL_PATH:-"./RoG-model"}

API_KEY=${API_KEY:-${OPENAI_API_KEY:-${SILICONFLOW_API_KEY:-}}}
API_BASE=${API_BASE:-"https://api.siliconflow.cn/v1"}
POISON_MODEL_NAME=${POISON_MODEL_NAME:-"deepseek-ai/DeepSeek-V3.2"}
API_TIMEOUT=${API_TIMEOUT:-"60"}
API_MAX_RETRIES=${API_MAX_RETRIES:-"3"}
API_MAX_TOKENS=${API_MAX_TOKENS:-"256"}
REQUIRE_API=${REQUIRE_API:-"0"}
MAX_FALLBACK_API_ERROR=${MAX_FALLBACK_API_ERROR:-"100"}
RESUME=${RESUME:-"0"}
SAVE_EVERY=${SAVE_EVERY:-"1"}

NUM_CANDIDATES=${NUM_CANDIDATES:-"5"}
INJECT_TOP_K=${INJECT_TOP_K:-"5"}
BUDGET_K=${BUDGET_K:-"20"}
HOP_REPEAT=${HOP_REPEAT:-"1"}
SINGLE_HOP_REPEAT=${SINGLE_HOP_REPEAT:-"1"}
FRONT_BOOST=${FRONT_BOOST:-"1.0"}
TYPE_MATCH_BOOST=${TYPE_MATCH_BOOST:-"1.0"}
TYPE_MISMATCH_BOOST=${TYPE_MISMATCH_BOOST:-"1.0"}

PROMPT_PATH=${PROMPT_PATH:-"prompts/llama2_predict.txt"}
PRED_ROOT=${PRED_ROOT:-"results/KGQA"}
API_USAGE_REPORT=${API_USAGE_REPORT:-"results/evaluation/api_usage/ragsafety_poison_api_usage.json"}
EVAL_REPORT=${EVAL_REPORT:-"results/evaluation/cascade_eval_report_ragsafety.json"}
RUN_CONFIG_OUT=${RUN_CONFIG_OUT:-"${POISONED_DATASET_PATH}.run_config.json"}

run_stage() {
  local stage="$1"
  [[ ",${STAGES}," == *",${stage},"* ]]
}

mkdir -p "$(dirname "$EVAL_REPORT")"
mkdir -p "$(dirname "$API_USAGE_REPORT")"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[error] MODEL_PATH does not look like a local model directory: ${MODEL_PATH}" >&2
  echo "        gen_rule_path.py uses local_files_only=True, so RoG weights must exist locally." >&2
  exit 1
fi

if [[ -z "${API_KEY}" ]]; then
  read -r -s -p "SiliconFlow API key: " API_KEY
  printf "\n"
fi

if run_stage decompose && [[ "$RUN_DECOMPOSE" == "1" ]]; then
  printf "\n[0/4] Decompose questions to sub-questions (RAG Safety aligned)...\n"
  python src/attack_scripts_adaptive/decompose_subquestions.py \
    --input_file "$ORIGINAL_DATASET_PATH" \
    --output_file "$DATASET_PATH"
fi

if run_stage rule; then
  printf "\n[1/4] Generate rules for sub-questions...\n"
  python src/qa_prediction/gen_rule_path.py \
    --d "$DATASET_PATH" \
    --split test \
    --model_name RoG \
    --model_path "$MODEL_PATH" \
    --n_beam 3 \
    --output_path "$RULE_OUTPUT_ROOT" \
    --force
fi

if run_stage poison; then
  printf "\n[2/4] Inject RAG Safety aligned poison triples with DeepSeek...\n"
  cmd=(
    python src/attack_scripts_adaptive/poison_data_adaptive.py
    --input_file "$DATASET_PATH"
    --rule_file "$RULE_FILE"
    --output_file "$POISONED_DATASET_PATH"
    --api_base "$API_BASE"
    --model_name "$POISON_MODEL_NAME"
    --api_timeout "$API_TIMEOUT"
    --api_max_retries "$API_MAX_RETRIES"
    --api_max_tokens "$API_MAX_TOKENS"
    --max_fallback_api_error "$MAX_FALLBACK_API_ERROR"
    --save_every "$SAVE_EVERY"
    --num_candidates "$NUM_CANDIDATES"
    --inject_top_k "$INJECT_TOP_K"
    --budget_k "$BUDGET_K"
    --hop_repeat "$HOP_REPEAT"
    --single_hop_repeat "$SINGLE_HOP_REPEAT"
    --front_boost "$FRONT_BOOST"
    --hop_boost_if_type_match "$TYPE_MATCH_BOOST"
    --hop_boost_if_type_mismatch "$TYPE_MISMATCH_BOOST"
    --api_usage_report "$API_USAGE_REPORT"
    --run_config_out "$RUN_CONFIG_OUT"
  )

  if [[ -n "${API_KEY}" ]]; then
    cmd+=(--api_key "$API_KEY")
  fi

  if [[ "${REQUIRE_API}" == "1" ]]; then
    cmd+=(--require_api)
  fi

  if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
  fi

  "${cmd[@]}"
fi

if run_stage predict; then
  printf "\n[3/4] Run clean inference...\n"
  python src/qa_prediction/predict_answer.py \
    --data_path "$DATASET_PATH" \
    --d cascade-clean-ragsafety \
    --split test \
    --model_name RoG \
    --model_path "$MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --add_rule \
    --rule_path "$RULE_FILE" \
    --predict_path "$PRED_ROOT" \
    --force

  printf "\n[3/4] Run poisoned inference...\n"
  python src/qa_prediction/predict_answer.py \
    --data_path "$POISONED_DATASET_PATH" \
    --d cascade-poison-ragsafety \
    --split test \
    --model_name RoG \
    --model_path "$MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --add_rule \
    --rule_path "$RULE_FILE" \
    --predict_path "$PRED_ROOT" \
    --force
fi

CLEAN_PRED_FILE="${PRED_ROOT}/cascade-clean-ragsafety/RoG/test/$(echo "$RULE_FILE" | tr '/.' '__')/predictions.jsonl"
POISON_PRED_FILE="${PRED_ROOT}/cascade-poison-ragsafety/RoG/test/$(echo "$RULE_FILE" | tr '/.' '__')/predictions.jsonl"

if run_stage eval; then
  printf "\n[4/4] Evaluate clean ACC + poisoned ASR/A-H@1...\n"
  python src/evaluation/eval_cascade.py \
    --clean_pred_file "$CLEAN_PRED_FILE" \
    --poison_pred_file "$POISON_PRED_FILE" \
    --poison_data_file "$POISONED_DATASET_PATH" \
    --report_file "$EVAL_REPORT"
fi

printf "\nDone. RAG Safety aligned report: %s\n" "$EVAL_REPORT"
