#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned Table-3 pipeline (Clean / Rand / Ours) for CWQ + WebQSP.
# Default protocol: NO --cascade_mode.
#
# Usage:
#   bash scripts/run_table3_paper_aligned.sh
#   MODEL_PATH=/path/to/your/RoG-model bash scripts/run_table3_paper_aligned.sh
#   RULE_MODEL_PATH=./RoG-model PRED_MODEL_PATH=./Llama-2-7b-chat-hf bash scripts/run_table3_paper_aligned.sh
#   DATASETS=cwq STAGES=rule,poison,predict,table3 bash scripts/run_table3_paper_aligned.sh
#
# Advanced staged execution:
#   STAGES=rule,poison_rand,predict_clean,predict_rand,table3
#   STAGES=diagnose,table3

STAGES=${STAGES:-"rule,poison,predict,table3"}
DATASETS=${DATASETS:-"cwq,webqsp"}

MODEL_NAME=${MODEL_NAME:-RoG}
MODEL_PATH=${MODEL_PATH:-./RoG-model}
RULE_MODEL_PATH=${RULE_MODEL_PATH:-${MODEL_PATH}}
PRED_MODEL_PATH=${PRED_MODEL_PATH:-${MODEL_PATH}}
PROMPT_PATH=${PROMPT_PATH:-prompts/llama2_predict.txt}
N_BEAM=${N_BEAM:-3}

# Separate attack-strength knobs for rand/ours.
# Rand defaults are intentionally mild to prevent over-poisoning collapse.
RAND_HOP_REPEAT=${RAND_HOP_REPEAT:-3}
RAND_SINGLE_HOP_REPEAT=${RAND_SINGLE_HOP_REPEAT:-6}
RAND_FRONT_BOOST=${RAND_FRONT_BOOST:-1.0}
RAND_HOP_BOOST_MATCH=${RAND_HOP_BOOST_MATCH:-1.0}
RAND_HOP_BOOST_MISMATCH=${RAND_HOP_BOOST_MISMATCH:-1.0}
RAND_INJECT_TOP_K=${RAND_INJECT_TOP_K:-1}

OURS_HOP_REPEAT=${OURS_HOP_REPEAT:-100}
OURS_SINGLE_HOP_REPEAT=${OURS_SINGLE_HOP_REPEAT:-200}
OURS_FRONT_BOOST=${OURS_FRONT_BOOST:-1.4}
OURS_HOP_BOOST_MATCH=${OURS_HOP_BOOST_MATCH:-1.5}
OURS_HOP_BOOST_MISMATCH=${OURS_HOP_BOOST_MISMATCH:-0.7}
OURS_INJECT_TOP_K=${OURS_INJECT_TOP_K:-3}

RULE_ROOT=${RULE_ROOT:-results/gen_rule_path}
PRED_ROOT=${PRED_ROOT:-results/KGQA}
EVAL_ROOT=${EVAL_ROOT:-results/evaluation}
TABLE3_COLLECT_SCRIPT=${TABLE3_COLLECT_SCRIPT:-scripts/table3_collect.py}
TABLE3_PREFER_DETAILED_EVAL=${TABLE3_PREFER_DETAILED_EVAL:-1}

CWQ_CLEAN=${CWQ_CLEAN:-datasets/clean_cwq.jsonl}
WEBQSP_CLEAN=${WEBQSP_CLEAN:-datasets/clean_webqsp.jsonl}
CWQ_PARQUET_GLOB=${CWQ_PARQUET_GLOB:-datasets/cwq/test-*.parquet}
WEBQSP_PARQUET_GLOB=${WEBQSP_PARQUET_GLOB:-datasets/webqsp/test-*.parquet}
# Ours usually uses decomposed set (e.g. cwq_full/webqsp_full).
OURS_CWQ_BASE=${OURS_CWQ_BASE:-datasets/cwq_full.jsonl}
OURS_WEBQSP_BASE=${OURS_WEBQSP_BASE:-datasets/webqsp_full.jsonl}
OURS_ALLOW_CLEAN_FALLBACK=${OURS_ALLOW_CLEAN_FALLBACK:-0}

# Optional official test-set aliases (from HF snapshots / converted exports).
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

dataset_rand_base_file() {
  local d="$1"
  dataset_clean_file "$d"
}

dataset_ours_base_file() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  if [[ "$d" == "cwq" ]]; then
    if [[ -f "${OURS_CWQ_BASE}" ]]; then
      echo "${OURS_CWQ_BASE}"
    else
      if [[ "${OURS_ALLOW_CLEAN_FALLBACK}" == "1" ]]; then
        echo "${clean_file}"
      else
        echo "${OURS_CWQ_BASE}"
      fi
    fi
  else
    if [[ -f "${OURS_WEBQSP_BASE}" ]]; then
      echo "${OURS_WEBQSP_BASE}"
    else
      if [[ "${OURS_ALLOW_CLEAN_FALLBACK}" == "1" ]]; then
        echo "${clean_file}"
      else
        echo "${OURS_WEBQSP_BASE}"
      fi
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

  [[ -f "${output_file}" ]] && return 0

  local parquet_files=()
  mapfile -t parquet_files < <(compgen -G "${glob_pattern}" || true)
  (( ${#parquet_files[@]} == 0 )) && return 0

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
  local data_file="$1"
  local data_tag
  data_tag="$(basename "$data_file" .jsonl)"
  echo "${RULE_ROOT}/${data_tag}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl"
}

rule_file_clean() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  rule_file "$clean_file"
}

rule_file_ours() {
  local d="$1"
  local ours_file
  ours_file="$(dataset_ours_base_file "$d")"
  rule_file "$ours_file"
}

rule_stage() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  python src/qa_prediction/gen_rule_path.py \
    --d "$clean_file" \
    --split test \
    --model_name "$MODEL_NAME" \
    --model_path "$RULE_MODEL_PATH" \
    --n_beam "$N_BEAM" \
    --output_path "$RULE_ROOT" \
    --force
}

rule_stage_ours() {
  local d="$1"
  local ours_file clean_file
  ours_file="$(dataset_ours_base_file "$d")"
  clean_file="$(dataset_clean_file "$d")"
  # If ours base equals clean base, avoid duplicate generation.
  if [[ "$ours_file" == "$clean_file" ]]; then
    return 0
  fi
  python src/qa_prediction/gen_rule_path.py \
    --d "$ours_file" \
    --split test \
    --model_name "$MODEL_NAME" \
    --model_path "$RULE_MODEL_PATH" \
    --n_beam "$N_BEAM" \
    --output_path "$RULE_ROOT" \
    --force
}

poison_rand_stage() {
  local d="$1"
  local rand_file
  rand_file="$(dataset_rand_base_file "$d")"
  local rf
  rf="$(rule_file_clean "$d")"

  python src/attack_scripts_adaptive/poison_data_adaptive.py \
    --input_file "$rand_file" \
    --rule_file "$rf" \
    --output_file "datasets/poisoned_${d}_rand.jsonl" \
    --mode rand \
    --inject_top_k "$RAND_INJECT_TOP_K" \
    --hop_repeat "$RAND_HOP_REPEAT" \
    --single_hop_repeat "$RAND_SINGLE_HOP_REPEAT" \
    --front_boost "$RAND_FRONT_BOOST" \
    --hop_boost_if_type_match "$RAND_HOP_BOOST_MATCH" \
    --hop_boost_if_type_mismatch "$RAND_HOP_BOOST_MISMATCH"
}

poison_ours_stage() {
  local d="$1"
  local ours_file
  ours_file="$(dataset_ours_base_file "$d")"
  local rf
  rf="$(rule_file_ours "$d")"

  python src/attack_scripts_adaptive/poison_data_adaptive.py \
    --input_file "$ours_file" \
    --rule_file "$rf" \
    --output_file "datasets/poisoned_${d}_ours.jsonl" \
    --mode ours \
    --inject_top_k "$OURS_INJECT_TOP_K" \
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
    --model_path "$PRED_MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --add_rule \
    --rule_path "$rf" \
    --predict_path "$PRED_ROOT" \
    --force
}

predict_clean_stage() {
  local d="$1"
  local clean_file
  clean_file="$(dataset_clean_file "$d")"
  local rf
  rf="$(rule_file_clean "$d")"
  predict_once "$clean_file" "${d}-clean-paper" "$rf"
}

predict_rand_stage() {
  local d="$1"
  local rf
  rf="$(rule_file_clean "$d")"
  predict_once "datasets/poisoned_${d}_rand.jsonl" "${d}-rand-paper" "$rf"
}

predict_ours_stage() {
  local d="$1"
  local rf
  rf="$(rule_file_ours "$d")"
  predict_once "datasets/poisoned_${d}_ours.jsonl" "${d}-ours-paper" "$rf"
}

table3_stage() {
  local d="$1"
  local clean_rule_postfix ours_rule_postfix
  clean_rule_postfix="$(echo "$(rule_file_clean "$d")" | tr '/.' '__')"
  ours_rule_postfix="$(echo "$(rule_file_ours "$d")" | tr '/.' '__')"

  local dataset_name
  if [[ "$d" == "cwq" ]]; then
    dataset_name="CWQ"
  else
    dataset_name="WebQSP"
  fi

  local extra_collect_args=()
  if [[ "${TABLE3_PREFER_DETAILED_EVAL}" == "1" ]]; then
    extra_collect_args+=(--prefer_detailed_eval)
  fi

  python "$TABLE3_COLLECT_SCRIPT" \
    --dataset "$dataset_name" \
    --method "$MODEL_NAME" \
    --clean_pred "${PRED_ROOT}/${d}-clean-paper/${MODEL_NAME}/test/${clean_rule_postfix}/predictions.jsonl" \
    --rand_pred "${PRED_ROOT}/${d}-rand-paper/${MODEL_NAME}/test/${clean_rule_postfix}/predictions.jsonl" \
    --ours_pred "${PRED_ROOT}/${d}-ours-paper/${MODEL_NAME}/test/${ours_rule_postfix}/predictions.jsonl" \
    --output_csv "${EVAL_ROOT}/table3_rows.csv" \
    "${extra_collect_args[@]}"
}

diagnose_prediction_file() {
  local path="$1"
  local tag="$2"
  if [[ ! -f "$path" ]]; then
    echo "[diagnose] ${tag}: missing file: ${path}"
    return 0
  fi

  python - "$path" "$tag" <<'PY'
import json
import sys

path = sys.argv[1]
tag = sys.argv[2]
total = 0
empty_pred = 0
multiline = 0
json_error = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        try:
            item = json.loads(line)
        except Exception:
            json_error += 1
            continue
        pred = item.get("prediction", "")
        if isinstance(pred, list):
            pred_txt = "\n".join(str(x) for x in pred)
        else:
            pred_txt = str(pred)
        if not pred_txt.strip():
            empty_pred += 1
        if "\n" in pred_txt.strip():
            multiline += 1

print(f"[diagnose] {tag}: total={total}, empty_prediction={empty_pred}, multiline_prediction={multiline}, json_error={json_error}")
PY
}

diagnose_stage() {
  local d="$1"
  local clean_rule_postfix ours_rule_postfix
  clean_rule_postfix="$(echo "$(rule_file_clean "$d")" | tr '/.' '__')"
  ours_rule_postfix="$(echo "$(rule_file_ours "$d")" | tr '/.' '__')"
  local base="${PRED_ROOT}"
  local clean_path="${base}/${d}-clean-paper/${MODEL_NAME}/test/${clean_rule_postfix}/predictions.jsonl"
  local rand_path="${base}/${d}-rand-paper/${MODEL_NAME}/test/${clean_rule_postfix}/predictions.jsonl"
  local ours_path="${base}/${d}-ours-paper/${MODEL_NAME}/test/${ours_rule_postfix}/predictions.jsonl"
  diagnose_prediction_file "$clean_path" "${d}/clean"
  diagnose_prediction_file "$rand_path" "${d}/rand"
  diagnose_prediction_file "$ours_path" "${d}/ours"
}

validate_config() {
  if [[ "$MODEL_PATH" == *"DATASETS="* || "$MODEL_PATH" == *"STAGES="* ]]; then
    echo "[error] MODEL_PATH looks malformed: ${MODEL_PATH}" >&2
    echo "        It seems environment variables were concatenated." >&2
    exit 2
  fi

  if [[ "$RULE_MODEL_PATH" == *"DATASETS="* || "$RULE_MODEL_PATH" == *"STAGES="* ]]; then
    echo "[error] RULE_MODEL_PATH looks malformed: ${RULE_MODEL_PATH}" >&2
    exit 2
  fi

  if [[ "$PRED_MODEL_PATH" == *"DATASETS="* || "$PRED_MODEL_PATH" == *"STAGES="* ]]; then
    echo "[error] PRED_MODEL_PATH looks malformed: ${PRED_MODEL_PATH}" >&2
    exit 2
  fi

  if [[ ! -d "${RULE_MODEL_PATH}" ]]; then
    echo "[warn] RULE_MODEL_PATH does not look like a local model directory: ${RULE_MODEL_PATH}" >&2
    echo "       src/qa_prediction/gen_rule_path.py uses local_files_only=True, so weights must exist locally." >&2
  fi

  if [[ ! -d "${PRED_MODEL_PATH}" ]]; then
    echo "[warn] PRED_MODEL_PATH does not look like a local model directory: ${PRED_MODEL_PATH}" >&2
    echo "       src/qa_prediction/predict_answer.py LLM loading expects local model path for local inference." >&2
  fi

  if run_stage table3 && [[ ! -f "${TABLE3_COLLECT_SCRIPT}" ]]; then
    echo "[error] Missing table3 collector script: ${TABLE3_COLLECT_SCRIPT}" >&2
    echo "        Keep scripts/table3_collect.py, or remove table3 from STAGES." >&2
    exit 2
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

  local ours_file
  ours_file="$(dataset_ours_base_file "$d")"
  if [[ -f "$ours_file" ]]; then
    echo "[${d}] ours_base=${ours_file}"
  else
    if run_stage poison || run_stage poison_ours; then
      echo "[error] Missing ours_base for ${d}: ${ours_file}" >&2
      echo "        Set OURS_CWQ_BASE/OURS_WEBQSP_BASE correctly, or enable OURS_ALLOW_CLEAN_FALLBACK=1." >&2
      exit 2
    fi
    echo "[${d}] ours_base missing (poison_ours not scheduled): ${ours_file}"
  fi

  count="$(wc -l < "$clean_file" | tr -d ' ')"
  echo "[${d}] dataset=${clean_file} (#lines=${count}, expected≈${expected})"
  if [[ "$count" -lt $((expected - 200)) || "$count" -gt $((expected + 200)) ]]; then
    echo "[warn] ${d} sample count is far from paper test split; Clean/Rand may be non-comparable." >&2
  fi
  python - "$d" "$clean_file" <<'PY'
import json
import sys
from collections import Counter

d = sys.argv[1]
path = sys.argv[2]
prefix_counter = Counter()
total = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
        except Exception:
            continue
        qid = str(item.get("id", "")).strip()
        if not qid:
            continue
        total += 1
        if qid.startswith("WebQ"):
            prefix_counter["WebQ"] += 1
        elif qid.startswith("CWQ"):
            prefix_counter["CWQ"] += 1
        else:
            prefix_counter["Other"] += 1

if total == 0:
    print(f"[warn] {d} id-prefix check skipped: no valid ids in {path}", file=sys.stderr)
    raise SystemExit(0)

webq = prefix_counter["WebQ"]
cwq = prefix_counter["CWQ"]
other = prefix_counter["Other"]
print(f"[{d}] id-prefix summary: WebQ={webq}, CWQ={cwq}, Other={other}, Total={total}")

# NOTE:
# In many released CWQ exports for this project, ids can still start with "WebQ*".
# So for CWQ this prefix is only informational and should not trigger a hard warning.
if d == "webqsp" and cwq > 0:
    ratio = cwq / total * 100
    print(f"[warn] webqsp clean file contains CWQ-like ids ({cwq}/{total}, {ratio:.2f}%). Check dataset file mix-up.", file=sys.stderr)
PY
}


main() {
  mkdir -p "$RULE_ROOT" "$PRED_ROOT" "$EVAL_ROOT"
  echo "[config] MODEL_NAME=${MODEL_NAME}"
  echo "[config] RULE_MODEL_PATH=${RULE_MODEL_PATH}"
  echo "[config] PRED_MODEL_PATH=${PRED_MODEL_PATH}"
  validate_config

  echo "[config] MODEL_PATH=${MODEL_PATH}"
  echo "[config] DATASETS=${DATASETS}"
  echo "[config] STAGES=${STAGES}"
  echo "[config] OURS base cwq=${OURS_CWQ_BASE}, webqsp=${OURS_WEBQSP_BASE}, allow_clean_fallback=${OURS_ALLOW_CLEAN_FALLBACK}"
  echo "[config] TABLE3 prefer_detailed_eval=${TABLE3_PREFER_DETAILED_EVAL}"
  echo "[config] RAND topk=${RAND_INJECT_TOP_K}, repeat=${RAND_HOP_REPEAT}/${RAND_SINGLE_HOP_REPEAT}, front=${RAND_FRONT_BOOST}, match=${RAND_HOP_BOOST_MATCH}, mismatch=${RAND_HOP_BOOST_MISMATCH}"
  echo "[config] OURS topk=${OURS_INJECT_TOP_K}, repeat=${OURS_HOP_REPEAT}/${OURS_SINGLE_HOP_REPEAT}, front=${OURS_FRONT_BOOST}, match=${OURS_HOP_BOOST_MATCH}, mismatch=${OURS_HOP_BOOST_MISMATCH}"

  for d in cwq webqsp; do
    run_dataset "$d" || continue
    validate_dataset "$d"

    if run_stage rule; then
      echo "[$d] rule"
      rule_stage "$d"
      if run_stage poison || run_stage poison_ours || run_stage predict || run_stage predict_ours; then
        echo "[$d] rule (ours base)"
        rule_stage_ours "$d"
      fi
    fi

    if run_stage poison || run_stage poison_rand; then
      echo "[$d] poison rand"
      poison_rand_stage "$d"
    fi

    if run_stage poison || run_stage poison_ours; then
      echo "[$d] poison ours"
      poison_ours_stage "$d"
    fi

    if run_stage predict || run_stage predict_clean; then
      echo "[$d] predict clean"
      predict_clean_stage "$d"
    fi

    if run_stage predict || run_stage predict_rand; then
      echo "[$d] predict rand"
      predict_rand_stage "$d"
    fi

    if run_stage predict || run_stage predict_ours; then
      echo "[$d] predict ours"
      predict_ours_stage "$d"
    fi

    if run_stage table3; then
      echo "[$d] table3 collect"
      table3_stage "$d"
    fi

    if run_stage diagnose; then
      echo "[$d] diagnose prediction files"
      diagnose_stage "$d"
    fi
  done

  echo "Done. Table rows appended to: ${EVAL_ROOT}/table3_rows.csv"
}

main "$@"
