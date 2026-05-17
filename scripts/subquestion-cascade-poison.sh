#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Standard workflow for sub-question cascading poisoning:
# 0) (optional) decompose original questions into sub-questions
# 1) generate relation rules from decomposed sub-questions
# 2) inject multi-hop poison triples guided by generated rules
# 3) run clean / poisoned inference
# 4) report clean accuracy + attack manipulation metrics (A-Precision/A-H@1/A-MRR)

ORIGINAL_DATASET_PATH=${ORIGINAL_DATASET_PATH:-"datasets/clean_questions.jsonl"}
RUN_DECOMPOSE=${RUN_DECOMPOSE:-"1"}
DATASET_PATH=${DATASET_PATH:-"datasets/clean_subquestions.jsonl"}
RUN_RULE=${RUN_RULE:-"1"}
RUN_POISON=${RUN_POISON:-"1"}
RUN_CLEAN_PREDICT=${RUN_CLEAN_PREDICT:-"1"}
RUN_POISON_PREDICT=${RUN_POISON_PREDICT:-"1"}
RUN_EVAL=${RUN_EVAL:-"1"}
FORCE_PREDICT=${FORCE_PREDICT:-"1"}

RULE_OUTPUT_ROOT=${RULE_OUTPUT_ROOT:-"results/gen_rule_path"}
RULE_FILE=${RULE_FILE:-"${RULE_OUTPUT_ROOT}/clean_subquestions/RoG/test/predictions_3_False.jsonl"}
POISONED_DATASET_PATH=${POISONED_DATASET_PATH:-"datasets/poisoned_subquestions.jsonl"}
MODEL_PATH=${MODEL_PATH:-"./RoG-model"}
API_KEY=${API_KEY:-${OPENAI_API_KEY:-${SILICONFLOW_API_KEY:-}}}
API_BASE=${API_BASE:-"https://api.siliconflow.cn/v1"}
POISON_MODEL_NAME=${POISON_MODEL_NAME:-"deepseek-ai/DeepSeek-V3.2"}
API_TIMEOUT=${API_TIMEOUT:-"60"}
API_MAX_RETRIES=${API_MAX_RETRIES:-"3"}
API_MAX_TOKENS=${API_MAX_TOKENS:-"256"}
REQUIRE_API=${REQUIRE_API:-"0"}
MAX_FALLBACK_API_ERROR=${MAX_FALLBACK_API_ERROR:-"100"}
API_FAIL_FAST_THRESHOLD=${API_FAIL_FAST_THRESHOLD:-"20"}
RESUME=${RESUME:-"0"}
SAVE_EVERY=${SAVE_EVERY:-"1"}
NUM_CANDIDATES=${NUM_CANDIDATES:-"5"}
INJECT_TOP_K=${INJECT_TOP_K:-"5"}
TARGET_TOP_K=${TARGET_TOP_K:-"8"}
MULTI_HOP_INJECT_MODE=${MULTI_HOP_INJECT_MODE:-"first_then_second"}
HOP_REPEAT=${HOP_REPEAT:-"1"}
SINGLE_HOP_REPEAT=${SINGLE_HOP_REPEAT:-"1"}
FRONT_BOOST=${FRONT_BOOST:-"1.0"}
TYPE_MATCH_BOOST=${TYPE_MATCH_BOOST:-"1.0"}
TYPE_MISMATCH_BOOST=${TYPE_MISMATCH_BOOST:-"1.0"}
BUDGET_K=${BUDGET_K:-"0"}
PROMPT_PATH=${PROMPT_PATH:-"prompts/llama2_predict.txt"}
PRED_ROOT=${PRED_ROOT:-"results/KGQA"}
EVAL_REPORT=${EVAL_REPORT:-"results/evaluation/cascade_eval_report.json"}
API_USAGE_REPORT=${API_USAGE_REPORT:-"results/evaluation/api_usage/cascade_poison_api_usage.json"}
RUN_CONFIG_OUT=${RUN_CONFIG_OUT:-"${POISONED_DATASET_PATH}.run_config.json"}

mkdir -p "$(dirname "$EVAL_REPORT")"

if [[ "$RUN_DECOMPOSE" == "1" ]]; then
  printf "\n[0/4] Decompose questions to sub-questions...\n"
  python src/attack_scripts_adaptive/decompose_subquestions.py \
    --input_file "$ORIGINAL_DATASET_PATH" \
    --output_file "$DATASET_PATH"
fi

if [[ "$RUN_RULE" == "1" ]]; then
  printf "\n[1/4] Generate rules for sub-questions...\n"
  python src/qa_prediction/gen_rule_path.py \
    --d "$DATASET_PATH" \
    --split test \
    --model_name RoG \
    --model_path "$MODEL_PATH" \
    --n_beam 3 \
    --output_path "$RULE_OUTPUT_ROOT" \
    --force
else
  printf "\n[1/4] Skip rule generation. Using existing rule file: %s\n" "$RULE_FILE"
fi

if [[ "$RUN_POISON" == "1" ]]; then
  printf "\n[2/4] Inject adaptive multi-hop poison triples...\n"
  poison_cmd=(
    python src/attack_scripts_adaptive/poison_data_adaptive.py
    --input_file "$DATASET_PATH"
    --rule_file "$RULE_FILE"
    --model_name "$POISON_MODEL_NAME"
    --api_base "$API_BASE"
    --api_timeout "$API_TIMEOUT"
    --api_max_retries "$API_MAX_RETRIES"
    --api_max_tokens "$API_MAX_TOKENS"
    --max_fallback_api_error "$MAX_FALLBACK_API_ERROR"
    --api_fail_fast_threshold "$API_FAIL_FAST_THRESHOLD"
    --save_every "$SAVE_EVERY"
    --num_candidates "$NUM_CANDIDATES"
    --inject_top_k "$INJECT_TOP_K"
    --target_top_k "$TARGET_TOP_K"
    --multi_hop_inject_mode "$MULTI_HOP_INJECT_MODE"
    --hop_repeat "$HOP_REPEAT"
    --single_hop_repeat "$SINGLE_HOP_REPEAT"
    --front_boost "$FRONT_BOOST"
    --hop_boost_if_type_match "$TYPE_MATCH_BOOST"
    --hop_boost_if_type_mismatch "$TYPE_MISMATCH_BOOST"
    --budget_k "$BUDGET_K"
    --api_usage_report "$API_USAGE_REPORT"
    --run_config_out "$RUN_CONFIG_OUT"
    --output_file "$POISONED_DATASET_PATH"
  )
  if [[ -n "$API_KEY" ]]; then
    poison_cmd+=(--api_key "$API_KEY")
  fi
  if [[ "$REQUIRE_API" == "1" ]]; then
    poison_cmd+=(--require_api)
  fi
  if [[ "$RESUME" == "1" ]]; then
    poison_cmd+=(--resume)
  fi
  "${poison_cmd[@]}"
else
  printf "\n[2/4] Skip poison injection. Using existing poisoned file: %s\n" "$POISONED_DATASET_PATH"
fi

predict_force_args=()
if [[ "$FORCE_PREDICT" == "1" ]]; then
  predict_force_args+=(--force)
fi

if [[ "$RUN_CLEAN_PREDICT" == "1" ]]; then
  printf "\n[3/4] Run clean inference...\n"
  python src/qa_prediction/predict_answer.py \
    --data_path "$DATASET_PATH" \
    --d cascade-clean \
    --split test \
    --model_name RoG \
    --model_path "$MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --add_rule \
    --rule_path "$RULE_FILE" \
    --predict_path "$PRED_ROOT" \
    "${predict_force_args[@]}"
else
  printf "\n[3/4] Skip clean inference.\n"
fi

if [[ "$RUN_POISON_PREDICT" == "1" ]]; then
  printf "\n[3/4] Run poisoned inference...\n"
  python src/qa_prediction/predict_answer.py \
    --data_path "$POISONED_DATASET_PATH" \
    --d cascade-poison \
    --split test \
    --model_name RoG \
    --model_path "$MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --add_rule \
    --rule_path "$RULE_FILE" \
    --predict_path "$PRED_ROOT" \
    "${predict_force_args[@]}"
else
  printf "\n[3/4] Skip poisoned inference.\n"
fi

CLEAN_PRED_FILE="${PRED_ROOT}/cascade-clean/RoG/test/$(echo "$RULE_FILE" | tr '/.' '__')/predictions.jsonl"
POISON_PRED_FILE="${PRED_ROOT}/cascade-poison/RoG/test/$(echo "$RULE_FILE" | tr '/.' '__')/predictions.jsonl"

if [[ "$RUN_EVAL" == "1" ]]; then
  printf "\n[4/4] Evaluate clean ACC + poisoned ASR/A-H@1...\n"
  python src/evaluation/eval_cascade.py \
    --clean_pred_file "$CLEAN_PRED_FILE" \
    --poison_pred_file "$POISON_PRED_FILE" \
    --poison_data_file "$POISONED_DATASET_PATH" \
    --report_file "$EVAL_REPORT"
else
  printf "\n[4/4] Skip evaluation.\n"
fi

printf "\nDone. Report: %s\n" "$EVAL_REPORT"
