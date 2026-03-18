#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned Table-3 pipeline (Clean / Rand / Ours) for CWQ + WebQSP.
# Default protocol: NO --cascade_mode.
#
# Usage:
#   bash scripts/run_table3_paper_aligned.sh
#   MODEL_PATH=/path/to/your/RoG-model bash scripts/run_table3_paper_aligned.sh
#   DATASETS=cwq STAGES=rule,poison,predict,table3 bash scripts/run_table3_paper_aligned.sh

STAGES=${STAGES:-"rule,poison,predict,table3"}
DATASETS=${DATASETS:-"cwq,webqsp"}

MODEL_NAME=${MODEL_NAME:-RoG}
MODEL_PATH=${MODEL_PATH:-./RoG-model}
PROMPT_PATH=${PROMPT_PATH:-prompts/llama2_predict.txt}
N_BEAM=${N_BEAM:-3}

# Separate attack-strength knobs for rand/ours (paper-aligned ablation convenience)
RAND_HOP_REPEAT=${RAND_HOP_REPEAT:-20}
RAND_SINGLE_HOP_REPEAT=${RAND_SINGLE_HOP_REPEAT:-40}
RAND_FRONT_BOOST=${RAND_FRONT_BOOST:-1.1}
RAND_HOP_BOOST_MATCH=${RAND_HOP_BOOST_MATCH:-1.2}
RAND_HOP_BOOST_MISMATCH=${RAND_HOP_BOOST_MISMATCH:-0.9}

OURS_HOP_REPEAT=${OURS_HOP_REPEAT:-100}
OURS_SINGLE_HOP_REPEAT=${OURS_SINGLE_HOP_REPEAT:-200}
OURS_FRONT_BOOST=${OURS_FRONT_BOOST:-1.4}
OURS_HOP_BOOST_MATCH=${OURS_HOP_BOOST_MATCH:-1.5}
OURS_HOP_BOOST_MISMATCH=${OURS_HOP_BOOST_MISMATCH:-0.7}

RULE_ROOT=${RULE_ROOT:-results/gen_rule_path}
PRED_ROOT=${PRED_ROOT:-results/KGQA}
EVAL_ROOT=${EVAL_ROOT:-results/evaluation}

CWQ_CLEAN=${CWQ_CLEAN:-datasets/clean_cwq.jsonl}
WEBQSP_CLEAN=${WEBQSP_CLEAN:-datasets/clean_webqsp.jsonl}

run_stage() {
  local stage="$1"
  [[ ",${STAGES}," == *",${stage},"* ]]
}

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

rule_file() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  local clean_tag
  clean_tag="$(basename "$clean_file" .jsonl)"
  echo "${RULE_ROOT}/${clean_tag}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl"
}

rule_stage() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  python src/qa_prediction/gen_rule_path.py \
    --d "$clean_file" \
    --split test \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --n_beam "$N_BEAM" \
    --output_path "$RULE_ROOT" \
    --force
}

poison_stage() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  local rf
  rf="$(rule_file "$d")"

  # Rand attacker (default weakened to avoid floor-effect saturation)
  python src/attack_scripts_adaptive/poison_data_adaptive.py \
    --input_file "$clean_file" \
    --rule_file "$rf" \
    --output_file "datasets/poisoned_${d}_rand.jsonl" \
    --mode rand \
    --hop_repeat "$RAND_HOP_REPEAT" \
    --single_hop_repeat "$RAND_SINGLE_HOP_REPEAT" \
    --front_boost "$RAND_FRONT_BOOST" \
    --hop_boost_if_type_match "$RAND_HOP_BOOST_MATCH" \
    --hop_boost_if_type_mismatch "$RAND_HOP_BOOST_MISMATCH"

  # Ours attacker (keeps stronger defaults)
  python src/attack_scripts_adaptive/poison_data_adaptive.py \
    --input_file "$clean_file" \
    --rule_file "$rf" \
    --output_file "datasets/poisoned_${d}_ours.jsonl" \
    --mode ours \
    --hop_repeat "$OURS_HOP_REPEAT" \
    --single_hop_repeat "$OURS_SINGLE_HOP_REPEAT" \
    --front_boost "$OURS_FRONT_BOOST" \
    --hop_boost_if_type_match "$OURS_HOP_BOOST_MATCH" \
    --hop_boost_if_type_mismatch "$OURS_HOP_BOOST_MISMATCH"
}

predict_once() {
  local data_path="$1"
  local d_name="$2"
  local rf="$3"

  python src/qa_prediction/predict_answer.py \
    --data_path "$data_path" \
    --d "$d_name" \
    --split test \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --add_rule \
    --rule_path "$rf" \
    --predict_path "$PRED_ROOT" \
    --force
}

predict_stage() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  local rf
  rf="$(rule_file "$d")"

  predict_once "$clean_file" "${d}-clean-paper" "$rf"
  predict_once "datasets/poisoned_${d}_rand.jsonl" "${d}-rand-paper" "$rf"
  predict_once "datasets/poisoned_${d}_ours.jsonl" "${d}-ours-paper" "$rf"
}

table3_stage() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  local rule_postfix
  rule_postfix="$(echo "$(rule_file "$d")" | tr '/.' '__')"

  local dataset_name
  if [[ "$d" == "cwq" ]]; then
    dataset_name="CWQ"
  else
    dataset_name="WebQSP"
  fi

  python scripts/table3_collect.py \
    --dataset "$dataset_name" \
    --method "$MODEL_NAME" \
    --clean_pred "${PRED_ROOT}/${d}-clean-paper/${MODEL_NAME}/test/${rule_postfix}/predictions.jsonl" \
    --rand_pred "${PRED_ROOT}/${d}-rand-paper/${MODEL_NAME}/test/${rule_postfix}/predictions.jsonl" \
    --ours_pred "${PRED_ROOT}/${d}-ours-paper/${MODEL_NAME}/test/${rule_postfix}/predictions.jsonl" \
    --output_csv "${EVAL_ROOT}/table3_rows.csv"
}


validate_config() {
  # Common shell typo guard: accidentally concatenating env vars on one line,
  # e.g. MODEL_PATH=/path/to/modelDATASETS=cwq
  if [[ "$MODEL_PATH" == *"DATASETS="* || "$MODEL_PATH" == *"STAGES="* ]]; then
    echo "[error] MODEL_PATH looks malformed: ${MODEL_PATH}" >&2
    echo "        It seems environment variables were concatenated." >&2
    echo "        Correct usage examples:" >&2
    echo "          MODEL_PATH=/abs/path/to/RoG-model DATASETS=cwq bash scripts/run_table3_paper_aligned.sh" >&2
    echo "          export MODEL_PATH=/abs/path/to/RoG-model; export DATASETS=cwq; bash scripts/run_table3_paper_aligned.sh" >&2
    exit 2
  fi
}

main() {
  mkdir -p "$RULE_ROOT" "$PRED_ROOT" "$EVAL_ROOT"

  validate_config

  echo "[config] MODEL_PATH=${MODEL_PATH}"
  echo "[config] DATASETS=${DATASETS}"
  echo "[config] STAGES=${STAGES}"
  echo "[config] RAND repeat=${RAND_HOP_REPEAT}/${RAND_SINGLE_HOP_REPEAT}, front=${RAND_FRONT_BOOST}, match=${RAND_HOP_BOOST_MATCH}, mismatch=${RAND_HOP_BOOST_MISMATCH}"
  echo "[config] OURS repeat=${OURS_HOP_REPEAT}/${OURS_SINGLE_HOP_REPEAT}, front=${OURS_FRONT_BOOST}, match=${OURS_HOP_BOOST_MATCH}, mismatch=${OURS_HOP_BOOST_MISMATCH}"

  for d in cwq webqsp; do
    run_dataset "$d" || continue

    if run_stage rule; then
      echo "[$d] rule"
      rule_stage "$d"
    fi

    if run_stage poison; then
      echo "[$d] poison (rand + ours)"
      poison_stage "$d"
    fi

    if run_stage predict; then
      echo "[$d] predict (clean + rand + ours)"
      predict_stage "$d"
    fi

    if run_stage table3; then
      echo "[$d] table3 collect"
      table3_stage "$d"
    fi
  done

  echo "Done. Table rows appended to: ${EVAL_ROOT}/table3_rows.csv"
}

main "$@"
