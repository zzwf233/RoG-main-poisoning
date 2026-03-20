#!/usr/bin/env bash
set -euo pipefail

# One-click full pipeline for CWQ + WebQSP:
# 0) copy full clean files
# 1) decompose sub-questions
# 2) generate rule paths
# 3) adaptive poison
# 4) poison prediction
# 5) clean prediction
# 6) evaluation (+ shared-subquestion export)
#
# Usage examples:
#   bash scripts/run_full_cascade_benchmark.sh
#   STAGES=decompose,rule,poison_data bash scripts/run_full_cascade_benchmark.sh
#   DATASETS=cwq bash scripts/run_full_cascade_benchmark.sh

STAGES=${STAGES:-"copy,decompose,rule,poison_data,predict_poison,predict_clean,eval"}
DATASETS=${DATASETS:-"cwq,webqsp"}
MAX_SUBQUESTIONS=${MAX_SUBQUESTIONS:-3}
MODEL_NAME=${MODEL_NAME:-RoG}
MODEL_PATH=${MODEL_PATH:-./RoG-model}
PROMPT_PATH=${PROMPT_PATH:-prompts/llama2_predict.txt}
PREDICT_ROOT=${PREDICT_ROOT:-results/KGQA}
RULE_ROOT=${RULE_ROOT:-results/gen_rule_path}
EVAL_ROOT=${EVAL_ROOT:-results/evaluation}
N_BEAM=${N_BEAM:-3}
API_KEY=${API_KEY:-${OPENAI_API_KEY:-${SILICONFLOW_API_KEY:-}}}
API_BASE=${API_BASE:-https://api.siliconflow.cn/v1}
ATTACK_MODEL_NAME=${ATTACK_MODEL_NAME:-Qwen/Qwen2.5-VL-72B-Instruct}
ATTACK_TEMPERATURE=${ATTACK_TEMPERATURE:-0.7}
REQUIRE_API=${REQUIRE_API:-0}
API_USAGE_ROOT=${API_USAGE_ROOT:-results/evaluation/api_usage}
API_FAIL_FAST_THRESHOLD=${API_FAIL_FAST_THRESHOLD:-20}
NUM_CANDIDATES=${NUM_CANDIDATES:-5}
INJECT_TOP_K=${INJECT_TOP_K:-3}
API_TIMEOUT=${API_TIMEOUT:-5}
API_MAX_RETRIES=${API_MAX_RETRIES:-0}
API_MAX_TOKENS=${API_MAX_TOKENS:-128}
MULTI_HOP_INJECT_MODE=${MULTI_HOP_INJECT_MODE:-first_then_second}

if [[ -n "${API_KEY}" ]]; then
  export OPENAI_API_KEY="${API_KEY}"
  export SILICONFLOW_API_KEY="${API_KEY}"
  echo "[api] API key detected. LLM planner + GPT prediction (if selected) can call API."
else
  echo "[api] No API key detected. poison_data stage will fallback to entity-pool planner."
fi

run_stage() {
  local stage="$1"
  [[ ",${STAGES}," == *",${stage},"* ]]
}

run_dataset() {
  local d="$1"
  [[ ",${DATASETS}," == *",${d},"* ]]
}

copy_full_inputs() {
  if run_dataset cwq; then
    cp datasets/clean_cwq.jsonl datasets/cwq_full_raw.jsonl
    echo "[copy] datasets/clean_cwq.jsonl -> datasets/cwq_full_raw.jsonl"
  fi
  if run_dataset webqsp; then
    cp datasets/clean_webqsp.jsonl datasets/webqsp_full_raw.jsonl
    echo "[copy] datasets/clean_webqsp.jsonl -> datasets/webqsp_full_raw.jsonl"
  fi
}

decompose_dataset() {
  local name="$1"
  python src/attack_scripts_adaptive/decompose_subquestions.py \
    --input_file "datasets/${name}_full_raw.jsonl" \
    --output_file "datasets/${name}_full.jsonl" \
    --max_subquestions "${MAX_SUBQUESTIONS}"
}

gen_rule_dataset() {
  local name="$1"
  python src/qa_prediction/gen_rule_path.py \
    --d "datasets/${name}_full.jsonl" \
    --split test \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --n_beam "${N_BEAM}" \
    --output_path "${RULE_ROOT}" \
    --force
}

poison_dataset() {
  local name="$1"
  local api_report_path="${API_USAGE_ROOT}/${name}_poison_api_usage.json"
  local cmd=(
    python src/attack_scripts_adaptive/poison_data_adaptive.py
    --input_file "datasets/${name}_full.jsonl" \
    --rule_file "${RULE_ROOT}/${name}_full/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl" \
    --output_file "datasets/poisoned_${name}_dynamic_full.jsonl" \
    --api_base "${API_BASE}" \
    --model_name "${ATTACK_MODEL_NAME}" \
    --temperature "${ATTACK_TEMPERATURE}" \
    --num_candidates "${NUM_CANDIDATES}" \
    --inject_top_k "${INJECT_TOP_K}" \
    --api_timeout "${API_TIMEOUT}" \
    --api_max_retries "${API_MAX_RETRIES}" \
    --api_max_tokens "${API_MAX_TOKENS}" \
    --multi_hop_inject_mode "${MULTI_HOP_INJECT_MODE}" \
    --api_fail_fast_threshold "${API_FAIL_FAST_THRESHOLD}" \
    --api_usage_report "${api_report_path}" \
    --front_boost 1.4 \
    --hop_boost_if_type_match 1.5 \
    --hop_boost_if_type_mismatch 0.7
  )

  if [[ -n "${API_KEY}" ]]; then
    cmd+=(--api_key "${API_KEY}")
  fi
  if [[ "${REQUIRE_API}" == "1" ]]; then
    cmd+=(--require_api)
  fi

  "${cmd[@]}"
}


predict_dataset() {
  local split_tag="$1"   # clean or poison
  local name="$2"        # cwq / webqsp

  local data_path
  local d_name
  if [[ "$split_tag" == "poison" ]]; then
    data_path="datasets/poisoned_${name}_dynamic_full.jsonl"
    d_name="${name}-full-poison"
  else
    data_path="datasets/${name}_full.jsonl"
    d_name="${name}-full-clean"
  fi

  python src/qa_prediction/predict_answer.py \
    --data_path "${data_path}" \
    --d "${d_name}" \
    --split test \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --prompt_path "${PROMPT_PATH}" \
    --add_rule \
    --rule_path "${RULE_ROOT}/${name}_full/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl" \
    --predict_path "${PREDICT_ROOT}" \
    --force \
    --cascade_mode
}

eval_dataset() {
  local name="$1"
  local rule_postfix
  rule_postfix="$(echo "${RULE_ROOT}/${name}_full/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl" | tr '/.' '__')"

  local clean_pred="${PREDICT_ROOT}/${name}-full-clean/${MODEL_NAME}/test/${rule_postfix}/predictions.jsonl"
  local poison_pred="${PREDICT_ROOT}/${name}-full-poison/${MODEL_NAME}/test/${rule_postfix}/predictions.jsonl"

  python src/evaluation/eval_cascade.py \
    --clean_pred_file "${clean_pred}" \
    --poison_pred_file "${poison_pred}" \
    --poison_data_file "datasets/poisoned_${name}_dynamic_full.jsonl" \
    --report_file "${EVAL_ROOT}/${name}_cascade_eval_report.json" \
    --debug_spread \
    --shared_subquestions_file "${EVAL_ROOT}/${name}_shared_subquestions_119.jsonl"
}

main() {
  mkdir -p "${RULE_ROOT}" "${PREDICT_ROOT}" "${EVAL_ROOT}" "${API_USAGE_ROOT}"

  if run_stage copy; then
    copy_full_inputs
  fi

  for name in cwq webqsp; do
    run_dataset "$name" || continue

    if run_stage decompose; then
      echo "[${name}] decompose"
      decompose_dataset "$name"
    fi

    if run_stage rule; then
      echo "[${name}] gen_rule_path"
      gen_rule_dataset "$name"
    fi

    if run_stage poison_data; then
      echo "[${name}] poison_data_adaptive"
      poison_dataset "$name"
    fi

    if run_stage predict_poison; then
      echo "[${name}] predict poison"
      predict_dataset poison "$name"
    fi

    if run_stage predict_clean; then
      echo "[${name}] predict clean"
      predict_dataset clean "$name"
    fi

    if run_stage eval; then
      echo "[${name}] eval cascade"
      eval_dataset "$name"
    fi
  done
}

main "$@"
