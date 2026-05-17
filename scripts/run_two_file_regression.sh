#!/usr/bin/env bash
set -euo pipefail

# Reliable two-file ablation for this repository (no external upstream dependency).
#
# Matrix:
#   CC = current gen_rule_path + current predict_answer
#   BC = base    gen_rule_path + current predict_answer
#   CB = current gen_rule_path + base    predict_answer
#   BB = base    gen_rule_path + base    predict_answer
#
# "base" versions are loaded from git at BASE_REF (default: origin/master).
#
# Usage:
#   MODEL_PATH=/path/to/RoG-model DATA_FILE=datasets/clean_webqsp.jsonl \
#   bash scripts/run_two_file_regression.sh
#
# Optional:
#   BASE_REF=HEAD~1 SPLIT=test N_BEAM=3 MODEL_NAME=RoG FORCE=1 \
#   RULE_OUT=results/gen_rule_path_ablation PRED_OUT=results/KGQA_ablation \
#   CSV_OUT=results/KGQA_ablation/two_file_regression_summary.csv \
#   WORK_DIR=/tmp/rog_two_file_ablation KEEP_WORK_DIR=0 \
#   bash scripts/run_two_file_regression.sh
#
# Non-git directory fallback:
#   BASE_GEN_FILE=/path/to/base/gen_rule_path.py \
#   BASE_PRED_FILE=/path/to/base/predict_answer.py \
#   MODEL_PATH=/path/to/RoG-model DATA_FILE=/path/to/clean_webqsp.jsonl \
#   bash scripts/run_two_file_regression.sh

MODEL_NAME=${MODEL_NAME:-RoG}
MODEL_PATH=${MODEL_PATH:-./RoG-model}
DATA_FILE=${DATA_FILE:-datasets/clean_webqsp.jsonl}
SPLIT=${SPLIT:-test}
N_BEAM=${N_BEAM:-3}
RULE_OUT=${RULE_OUT:-results/gen_rule_path_ablation}
PRED_OUT=${PRED_OUT:-results/KGQA_ablation}
PROMPT_PATH=${PROMPT_PATH:-prompts/llama2_predict.txt}
CSV_OUT=${CSV_OUT:-${PRED_OUT}/two_file_regression_summary.csv}
WORK_DIR=${WORK_DIR:-/tmp/rog_two_file_ablation}
KEEP_WORK_DIR=${KEEP_WORK_DIR:-0}
FORCE=${FORCE:-1}
BASE_REF=${BASE_REF:-origin/master}
BASE_GEN_FILE=${BASE_GEN_FILE:-}
BASE_PRED_FILE=${BASE_PRED_FILE:-}

GEN_FILE=src/qa_prediction/gen_rule_path.py
PRED_FILE=src/qa_prediction/predict_answer.py

if [[ ! -f "${GEN_FILE}" || ! -f "${PRED_FILE}" ]]; then
  echo "[error] run from repo root: missing ${GEN_FILE} or ${PRED_FILE}" >&2
  exit 2
fi

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "[error] DATA_FILE not found: ${DATA_FILE}" >&2
  exit 2
fi

# Normalize to absolute paths (runner scripts can have different script_dir logic).
DATA_FILE="$(realpath "${DATA_FILE}")"
RULE_OUT="$(realpath -m "${RULE_OUT}")"
PRED_OUT="$(realpath -m "${PRED_OUT}")"
CSV_OUT="$(realpath -m "${CSV_OUT}")"
if [[ -f "${PROMPT_PATH}" ]]; then
  PROMPT_PATH="$(realpath "${PROMPT_PATH}")"
fi
if [[ -d "${MODEL_PATH}" || -f "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(realpath "${MODEL_PATH}")"
fi

mkdir -p "${WORK_DIR}" "${RULE_OUT}" "${PRED_OUT}" "$(dirname "${CSV_OUT}")"

resolve_base_ref() {
  local ref="$1"
  if git rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
    echo "${ref}"
    return 0
  fi

  local c
  for c in origin/master origin/main master main; do
    if git rev-parse --verify --quiet "${c}^{commit}" >/dev/null; then
      echo "${c}"
      return 0
    fi
  done

  # Fallback: previous commit on current HEAD (if exists).
  local prev
  prev="$(git rev-list --max-count=2 HEAD | tail -n1 || true)"
  if [[ -n "${prev}" ]] && git rev-parse --verify --quiet "${prev}^{commit}" >/dev/null; then
    echo "${prev}"
    return 0
  fi
  return 1
}

echo "[info] stage current runners -> ${WORK_DIR}"
cp "${GEN_FILE}" "${WORK_DIR}/gen_rule_path.current.py"
cp "${PRED_FILE}" "${WORK_DIR}/predict_answer.current.py"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! BASE_REF_RESOLVED="$(resolve_base_ref "${BASE_REF}")"; then
    echo "[error] BASE_REF is not valid and no fallback ref found: ${BASE_REF}" >&2
    echo "        Try one of: origin/master, origin/main, master, main, or a commit hash." >&2
    echo "        Recent commits:" >&2
    git --no-pager log --oneline -n 5 >&2 || true
    exit 2
  fi
  if [[ "${BASE_REF_RESOLVED}" != "${BASE_REF}" ]]; then
    echo "[warn] BASE_REF '${BASE_REF}' not found; fallback to '${BASE_REF_RESOLVED}'"
  fi
  BASE_REF="${BASE_REF_RESOLVED}"
  echo "[info] BASE_REF=${BASE_REF}"
  echo "[info] stage base runners from git ref ${BASE_REF} -> ${WORK_DIR}"
  git show "${BASE_REF}:${GEN_FILE}" > "${WORK_DIR}/gen_rule_path.base.py"
  git show "${BASE_REF}:${PRED_FILE}" > "${WORK_DIR}/predict_answer.base.py"
else
  echo "[warn] current directory is not a git repository; use BASE_GEN_FILE/BASE_PRED_FILE"
  if [[ -z "${BASE_GEN_FILE}" || -z "${BASE_PRED_FILE}" ]]; then
    echo "[error] missing BASE_GEN_FILE or BASE_PRED_FILE in non-git mode." >&2
    exit 2
  fi
  if [[ ! -f "${BASE_GEN_FILE}" || ! -f "${BASE_PRED_FILE}" ]]; then
    echo "[error] BASE_GEN_FILE or BASE_PRED_FILE path not found." >&2
    exit 2
  fi
  cp "${BASE_GEN_FILE}" "${WORK_DIR}/gen_rule_path.base.py"
  cp "${BASE_PRED_FILE}" "${WORK_DIR}/predict_answer.base.py"
fi

# Patch base runners for local JSONL compatibility (legacy loaders expect dataset scripts).
python - "${WORK_DIR}/gen_rule_path.base.py" "${WORK_DIR}/predict_answer.base.py" <<'PY'
import sys
from pathlib import Path

for p in map(Path, sys.argv[1:]):
    txt = p.read_text(encoding="utf-8")
    old = "dataset = load_dataset(input_file, split=args.split)"
    new = (
        "if os.path.isfile(input_file):\n"
        "        dataset = load_dataset('json', data_files=input_file, split='train')\n"
        "    else:\n"
        "        dataset = load_dataset(input_file, split=args.split)"
    )
    if old in txt:
        txt = txt.replace(old, new)

    old_truth = (
        "        paths = utils.get_truth_paths(sample[\"q_entity\"], sample[\"a_entity\"], graph)\n"
        "        ground_paths = set()\n"
        "        for path in paths:\n"
        "            ground_paths.add(tuple([p[1] for p in path]))  # extract relation path\n"
        "        sample[\"ground_paths\"] = list(ground_paths)\n"
        "        return sample"
    )
    new_truth = (
        "        try:\n"
        "            paths = utils.get_truth_paths(sample[\"q_entity\"], sample[\"a_entity\"], graph)\n"
        "            ground_paths = set()\n"
        "            for path in paths:\n"
        "                ground_paths.add(tuple([p[1] for p in path]))  # extract relation path\n"
        "            sample[\"ground_paths\"] = list(ground_paths)\n"
        "        except Exception:\n"
        "            sample[\"ground_paths\"] = []\n"
        "        return sample"
    )
    if old_truth in txt:
        txt = txt.replace(old_truth, new_truth)

    p.write_text(txt, encoding="utf-8")
PY

cleanup() {
  if [[ "${KEEP_WORK_DIR}" == "1" ]]; then
    echo "[info] keep work dir: ${WORK_DIR}"
  else
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT

run_case() {
  local case_name="$1"    # CC / BC / CB / BB
  local gen_src="$2"      # current / base
  local pred_src="$3"     # current / base

  echo
  echo "========== CASE ${case_name} (gen=${gen_src}, pred=${pred_src}) =========="

  local gen_runner="${WORK_DIR}/gen_rule_path.${gen_src}.py"
  local pred_runner="${WORK_DIR}/predict_answer.${pred_src}.py"
  [[ -f "${gen_runner}" && -f "${pred_runner}" ]] || {
    echo "[error] missing staged runner(s) for case ${case_name}" >&2
    exit 2
  }

  local force_flag=()
  if [[ "${FORCE}" == "1" ]]; then
    force_flag=(--force)
  fi

  local data_dir data_base
  data_dir="$(dirname "${DATA_FILE}")"
  data_base="$(basename "${DATA_FILE}")"

  if [[ "${gen_src}" == "base" ]]; then
    PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" python "${gen_runner}" \
      --data_path "${data_dir}" \
      --d "${data_base}" \
      --split "${SPLIT}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}" \
      --n_beam "${N_BEAM}" \
      --output_path "${RULE_OUT}" \
      "${force_flag[@]}"
  else
    PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" python "${gen_runner}" \
      --d "${DATA_FILE}" \
      --split "${SPLIT}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}" \
      --n_beam "${N_BEAM}" \
      --output_path "${RULE_OUT}" \
      "${force_flag[@]}"
  fi

  local data_tag
  data_tag="$(basename "${DATA_FILE}" .jsonl)"
  local rule_file="${RULE_OUT}/${data_tag}/${MODEL_NAME}/${SPLIT}/predictions_${N_BEAM}_False.jsonl"
  if [[ ! -f "${rule_file}" ]]; then
    local alt_rule_file="${RULE_OUT}/${data_base}/${MODEL_NAME}/${SPLIT}/predictions_${N_BEAM}_False.jsonl"
    if [[ -f "${alt_rule_file}" ]]; then
      rule_file="${alt_rule_file}"
    else
      echo "[error] missing rule file for ${case_name}:" >&2
      echo "        tried: ${rule_file}" >&2
      echo "        tried: ${alt_rule_file}" >&2
      exit 2
    fi
  fi

  local case_pred_root="${PRED_OUT}/${case_name}"
  mkdir -p "${case_pred_root}"
  local dname="${data_tag}-${case_name}"

  if [[ "${pred_src}" == "base" ]]; then
    PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" python "${pred_runner}" \
      --data_path "${data_dir}" \
      --d "${data_base}" \
      --split "${SPLIT}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}" \
      --prompt_path "${PROMPT_PATH}" \
      --add_rule \
      --rule_path "${rule_file}" \
      --predict_path "${case_pred_root}" \
      "${force_flag[@]}"
  else
    PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" python "${pred_runner}" \
      --data_path "${DATA_FILE}" \
      --d "${dname}" \
      --split "${SPLIT}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}" \
      --prompt_path "${PROMPT_PATH}" \
      --add_rule \
      --rule_path "${rule_file}" \
      --predict_path "${case_pred_root}" \
      "${force_flag[@]}"
  fi
}

run_case CC current current
run_case BC base current
run_case CB current base
run_case BB base base

echo
echo "[info] collect summary -> ${CSV_OUT}"
python - "${PRED_OUT}" "${CSV_OUT}" <<'PY'
import csv
import glob
import re
import sys

pred_root, csv_out = sys.argv[1:3]
cases = ["CC", "BC", "CB", "BB"]
pat = re.compile(
    r"Accuracy:\s*([0-9.]+)\s+Hit:\s*([0-9.]+)\s+F1:\s*([0-9.]+)\s+Precision:\s*([0-9.]+)\s+Recall:\s*([0-9.]+)"
)

rows = []
for c in cases:
    files = sorted(glob.glob(f"{pred_root}/{c}/**/eval_result.txt", recursive=True))
    if not files:
        rows.append({"case": c, "status": "missing_eval_result"})
        continue
    path = files[-1]
    txt = open(path, "r", encoding="utf-8").read().strip()
    m = pat.search(txt)
    if not m:
        rows.append({"case": c, "status": "parse_failed", "eval_result": txt, "path": path})
        continue
    acc, hit, f1, p, r = map(float, m.groups())
    rows.append({
        "case": c,
        "status": "ok",
        "accuracy": round(acc, 4),
        "hit": round(hit, 4),
        "f1": round(f1, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "eval_result_path": path,
    })

with open(csv_out, "w", encoding="utf-8", newline="") as f:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("Saved:", csv_out)
for r in rows:
    print(r)
PY

echo "[done] two-file regression completed."
