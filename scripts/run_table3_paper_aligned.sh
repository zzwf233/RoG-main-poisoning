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
API_KEY=${API_KEY:-${SILICONFLOW_API_KEY:-${OPENAI_API_KEY:-""}}}
API_BASE=${API_BASE:-"https://api.siliconflow.cn/v1"}
REQUIRE_API=${REQUIRE_API:-"0"}

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
CWQ_PARQUET_GLOB=${CWQ_PARQUET_GLOB:-datasets/cwq/test-*.parquet}
WEBQSP_PARQUET_GLOB=${WEBQSP_PARQUET_GLOB:-datasets/webqsp/test-*.parquet}

# Optional official test-set aliases (from HF snapshots / converted exports).
# If *_CLEAN does not exist but these do, we auto-fallback to reduce setup mistakes.
CWQ_PAPER_DEFAULT=${CWQ_PAPER_DEFAULT:-datasets/RoG-cwq_test.jsonl}
WEBQSP_PAPER_DEFAULT=${WEBQSP_PAPER_DEFAULT:-datasets/RoG-webqsp_test.jsonl}

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
    if [[ -f "${CWQ_CLEAN}" ]]; then
      echo "${CWQ_CLEAN}"
    elif [[ -f "datasets/clean_cwq_from_parquet.jsonl" ]]; then
      echo "datasets/clean_cwq_from_parquet.jsonl"
    elif [[ -f "${CWQ_PAPER_DEFAULT}" ]]; then
      echo "${CWQ_PAPER_DEFAULT}"
    else
      echo "${CWQ_CLEAN}"
    fi
  else
    if [[ -f "${WEBQSP_CLEAN}" ]]; then
      echo "${WEBQSP_CLEAN}"
    elif [[ -f "datasets/clean_webqsp_from_parquet.jsonl" ]]; then
      echo "datasets/clean_webqsp_from_parquet.jsonl"
    elif [[ -f "${WEBQSP_PAPER_DEFAULT}" ]]; then
      echo "${WEBQSP_PAPER_DEFAULT}"
    else
      echo "${WEBQSP_CLEAN}"
    fi
  fi
}

create_jsonl_from_parquet() {
  local output_file="$1"
  shift
  python - "$output_file" "$@" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj

out = Path(sys.argv[1])
inputs = sys.argv[2:]
if not inputs:
    raise SystemExit("No parquet inputs provided.")

dfs = [pd.read_parquet(p, engine="auto") for p in inputs]
df = pd.concat(dfs, ignore_index=True)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        sample = {
            "id": row.get("id", str(idx)),
            "question": row.get("question", ""),
            "graph": row.get("graph", []),
            "q_entity": row.get("q_entity", []),
            "a_entity": row.get("a_entity", []),
            "choices": row.get("choices", []),
            "answer": row.get("answer", []),
        }
        sample = make_serializable(sample)
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
print(f"[build] wrote {len(df)} rows -> {out}")
PY
}

build_clean_from_parquet_if_needed() {
  local d="$1"
  local glob_pattern output_file
  if [[ "$d" == "cwq" ]]; then
    [[ -f "${CWQ_CLEAN}" || -f "${CWQ_PAPER_DEFAULT}" ]] && return 0
    glob_pattern="${CWQ_PARQUET_GLOB}"
    output_file="datasets/clean_cwq_from_parquet.jsonl"
  else
    [[ -f "${WEBQSP_CLEAN}" || -f "${WEBQSP_PAPER_DEFAULT}" ]] && return 0
    glob_pattern="${WEBQSP_PARQUET_GLOB}"
    output_file="datasets/clean_webqsp_from_parquet.jsonl"
  fi

  if [[ -f "${output_file}" ]]; then
    return 0
  fi

  local parquet_files=()
  mapfile -t parquet_files < <(compgen -G "${glob_pattern}" || true)
  if (( ${#parquet_files[@]} == 0 )); then
    return 0
  fi

  echo "[${d}] build clean jsonl from parquet test shards (${#parquet_files[@]} files)"
  create_jsonl_from_parquet "${output_file}" "${parquet_files[@]}"
}

dataset_expected_size() {
  local d="$1"
  if [[ "$d" == "cwq" ]]; then
    echo "3531"
  else
    echo "1639"
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
    --api_key "$API_KEY" \
    --api_base "$API_BASE" \
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
    --api_key "$API_KEY" \
    --api_base "$API_BASE" \
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
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[warn] MODEL_PATH does not look like a local model directory: ${MODEL_PATH}" >&2
    echo "       src/qa_prediction/gen_rule_path.py uses local_files_only=True, so weights must exist locally." >&2
  fi
}

validate_dataset() {
  local d="$1"
  local clean_file expected count
  build_clean_from_parquet_if_needed "$d"
  clean_file="$(dataset_clean_file "$d")"
  expected="$(dataset_expected_size "$d")"

  if [[ ! -f "$clean_file" ]]; then
    echo "[error] Missing clean dataset for ${d}: ${clean_file}" >&2
    exit 2
  fi

  count="$(wc -l < "$clean_file" | tr -d ' ')"
  echo "[${d}] dataset=${clean_file} (#lines=${count}, expected≈${expected})"
  if [[ "$count" -lt $((expected - 200)) || "$count" -gt $((expected + 200)) ]]; then
    echo "[warn] ${d} sample count is far from paper test split; Clean/Rand may be non-comparable." >&2
  fi
}

main() {
  mkdir -p "$RULE_ROOT" "$PRED_ROOT" "$EVAL_ROOT"

  validate_config
  if [[ "$REQUIRE_API" == "1" && -z "$API_KEY" ]]; then
    echo "[error] REQUIRE_API=1 but no API key was provided." >&2
    echo "        Set API_KEY (or SILICONFLOW_API_KEY / OPENAI_API_KEY) before running." >&2
    exit 2
  fi
  echo "[config] MODEL_PATH=${MODEL_PATH}"
  echo "[config] DATASETS=${DATASETS}"
  echo "[config] STAGES=${STAGES}"
  echo "[config] RAND repeat=${RAND_HOP_REPEAT}/${RAND_SINGLE_HOP_REPEAT}, front=${RAND_FRONT_BOOST}, match=${RAND_HOP_BOOST_MATCH}, mismatch=${RAND_HOP_BOOST_MISMATCH}"
  echo "[config] API enabled=$([[ -n \"$API_KEY\" ]] && echo yes || echo no), API_BASE=${API_BASE}"
  echo "[config] OURS repeat=${OURS_HOP_REPEAT}/${OURS_SINGLE_HOP_REPEAT}, front=${OURS_FRONT_BOOST}, match=${OURS_HOP_BOOST_MATCH}, mismatch=${OURS_HOP_BOOST_MISMATCH}"

  for d in cwq webqsp; do
    run_dataset "$d" || continue
    validate_dataset "$d"

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
