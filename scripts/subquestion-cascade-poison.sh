#!/usr/bin/env bash
set -euo pipefail

# Standard workflow for sub-question cascading poisoning:
# 0) (optional) decompose original questions into sub-questions
# 1) generate relation rules from decomposed sub-questions
# 2) inject multi-hop poison triples guided by generated rules
# 3) run clean / poisoned inference
# 4) report clean accuracy + attack manipulation metrics (A-Precision/A-H@1/A-MRR)

ORIGINAL_DATASET_PATH=${ORIGINAL_DATASET_PATH:-"datasets/clean_questions.jsonl"}
RUN_DECOMPOSE=${RUN_DECOMPOSE:-"1"}
DATASET_PATH=${DATASET_PATH:-"datasets/clean_subquestions.jsonl"}

RULE_OUTPUT_ROOT=${RULE_OUTPUT_ROOT:-"results/gen_rule_path"}
RULE_FILE=${RULE_FILE:-"${RULE_OUTPUT_ROOT}/clean_subquestions/RoG/test/predictions_3_False.jsonl"}
POISONED_DATASET_PATH=${POISONED_DATASET_PATH:-"datasets/poisoned_subquestions.jsonl"}
MODEL_PATH=${MODEL_PATH:-"rmanluo/RoG"}
POISON_MODEL_NAME=${POISON_MODEL_NAME:-"Qwen/Qwen2.5-VL-72B-Instruct"}
FRONT_BOOST=${FRONT_BOOST:-"1.4"}
TYPE_MATCH_BOOST=${TYPE_MATCH_BOOST:-"1.5"}
TYPE_MISMATCH_BOOST=${TYPE_MISMATCH_BOOST:-"0.7"}
PROMPT_PATH=${PROMPT_PATH:-"prompts/llama2_predict.txt"}
PRED_ROOT=${PRED_ROOT:-"results/KGQA"}
EVAL_REPORT=${EVAL_REPORT:-"results/evaluation/cascade_eval_report.json"}

mkdir -p "$(dirname "$EVAL_REPORT")"

if [[ "$RUN_DECOMPOSE" == "1" ]]; then
  printf "\n[0/4] Decompose questions to sub-questions...\n"
  python src/attack_scripts_adaptive/decompose_subquestions.py \
    --input_file "$ORIGINAL_DATASET_PATH" \
    --output_file "$DATASET_PATH"
fi

printf "\n[1/4] Generate rules for sub-questions...\n"
python src/qa_prediction/gen_rule_path.py \
  --d "$DATASET_PATH" \
  --split test \
  --model_name RoG \
  --model_path "$MODEL_PATH" \
  --n_beam 3 \
  --output_path "$RULE_OUTPUT_ROOT" \
  --force

printf "\n[2/4] Inject adaptive multi-hop poison triples...\n"
python src/attack_scripts_adaptive/poison_data_adaptive.py \
  --input_file "$DATASET_PATH" \
  --rule_file "$RULE_FILE" \
  --model_name "$POISON_MODEL_NAME" \
  --front_boost "$FRONT_BOOST" \
  --hop_boost_if_type_match "$TYPE_MATCH_BOOST" \
  --hop_boost_if_type_mismatch "$TYPE_MISMATCH_BOOST" \
  --output_file "$POISONED_DATASET_PATH"

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
  --force

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
  --force

CLEAN_PRED_FILE="${PRED_ROOT}/cascade-clean/RoG/test/$(echo "$RULE_FILE" | tr '/.' '__')/predictions.jsonl"
POISON_PRED_FILE="${PRED_ROOT}/cascade-poison/RoG/test/$(echo "$RULE_FILE" | tr '/.' '__')/predictions.jsonl"

printf "\n[4/4] Evaluate clean ACC + poisoned ASR/A-H@1...\n"
python src/evaluation/eval_cascade.py \
  --clean_pred_file "$CLEAN_PRED_FILE" \
  --poison_pred_file "$POISON_PRED_FILE" \
  --poison_data_file "$POISONED_DATASET_PATH" \
  --report_file "$EVAL_REPORT"

printf "\nDone. Report: %s\n" "$EVAL_REPORT"
