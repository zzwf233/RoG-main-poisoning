#!/usr/bin/env python
import argparse
import json
import re
import string
from pathlib import Path


SKIP_LINE_PATTERNS = [
    r"^\s*sure\b",
    r"^\s*here\s+(are|is)\b",
    r"^\s*based on\b",
    r"^\s*the final answer",
    r"^\s*final answers?\b",
    r"^\s*i'?m sorry\b",
    r"^\s*i cannot\b",
    r"^\s*please\b",
]


def normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(<pad>)\b", " ", text)
    return " ".join(text.split())


def match(prediction: str, answer: str) -> bool:
    answer_norm = normalize(answer)
    return bool(answer_norm) and answer_norm in normalize(prediction)


def should_skip_line(line: str) -> bool:
    low = line.strip().lower()
    if not low:
        return True
    return any(re.search(pat, low) for pat in SKIP_LINE_PATTERNS)


def clean_candidate(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^[-*•]\s*", "", text)
    text = re.sub(r"^\d+[\).]\s*", "", text)
    text = re.sub(r"^(answer|answers|final answer|final answers)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip("\"'`").strip()

    if "->" in text:
        text = text.split("->")[-1].strip()

    # Common sentence forms from instruction-tuned Llama outputs.
    sentence_patterns = [
        r"^.*?\b(?:answer is|answers are)\s+(.+)$",
        r"^.*?\b(?:is|are|was|were)\s+(?:from|located in|played by|married to|spoken by)\s+(.+)$",
        r"^.*?\b(?:speak|speaks|spoke)\s+(.+)$",
        r"^.*?\b(?:invented|did)\s+(.+)$",
        r"^.*?\b(?:by actor|by)\s+(.+)$",
    ]
    for pat in sentence_patterns:
        m = re.match(pat, text, flags=re.I)
        if m:
            text = m.group(1).strip()
            break

    text = re.sub(r"\s+\(.*?\)\s*$", "", text).strip()
    text = text.strip(" .;:")
    return text


def split_inline_candidates(text: str):
    # Keep entity names with commas mostly intact; only split clear list separators.
    text = re.sub(r"\s+(?:and|or)\s+", "\n", text)
    if ";" in text:
        text = text.replace(";", "\n")
    return [x.strip() for x in text.splitlines() if x.strip()]


def parse_prediction(prediction):
    if isinstance(prediction, list):
        raw_lines = [str(x) for x in prediction]
    else:
        raw = str(prediction)
        raw = raw.replace("\\n", "\n")
        raw_lines = raw.splitlines()

    candidates = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if should_skip_line(line):
            continue
        for part in split_inline_candidates(line):
            cand = clean_candidate(part)
            if not cand:
                continue
            if should_skip_line(cand):
                continue
            # Drop leftovers that are still instructions rather than answers.
            if len(cand.split()) > 18 and not any(ch.isupper() for ch in cand):
                continue
            candidates.append(cand)

    if not candidates and raw_lines:
        fallback = clean_candidate(raw_lines[-1])
        if fallback:
            candidates.append(fallback)

    deduped = []
    seen = set()
    for cand in candidates:
        key = normalize(cand)
        if key and key not in seen:
            seen.add(key)
            deduped.append(cand)
    return deduped


def eval_f1(predictions, answers):
    if not predictions:
        return 0.0, 0.0, 0.0
    prediction_str = " ".join(predictions)
    matched = sum(1 for ans in answers if match(prediction_str, ans))
    precision = matched / len(predictions)
    recall = matched / len(answers) if answers else 0.0
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def eval_hits1(predictions, answers):
    if not predictions:
        return 0
    return int(any(match(predictions[0], ans) for ans in answers))


def eval_hit(predictions, answers):
    prediction_str = " ".join(predictions)
    return int(any(match(prediction_str, ans) for ans in answers))


def eval_acc(predictions, answers):
    if not answers:
        return 0.0
    prediction_str = " ".join(predictions)
    return sum(1 for ans in answers if match(prediction_str, ans)) / len(answers)


def eval_em(predictions, answers):
    pred_set = {normalize(pred) for pred in predictions if normalize(pred)}
    ans_set = {normalize(ans) for ans in answers if normalize(ans)}
    return int(pred_set == ans_set)


def evaluate(prediction_file: Path, output_prefix: Path):
    parsed_file = output_prefix.with_suffix(".parsed.jsonl")
    detailed_file = output_prefix.with_suffix(".detailed_eval_result_parsed.jsonl")
    result_file = output_prefix.with_suffix(".eval_result_parsed.txt")

    totals = {
        "hit": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "hits1": [],
        "em": [],
        "acc": [],
    }

    with prediction_file.open("r", encoding="utf-8") as fin, \
            parsed_file.open("w", encoding="utf-8") as fparsed, \
            detailed_file.open("w", encoding="utf-8") as fdetailed:
        for line in fin:
            row = json.loads(line)
            answers = row.get("ground_truth", row.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]
            parsed = parse_prediction(row.get("prediction", ""))

            parsed_row = dict(row)
            parsed_row["raw_prediction"] = row.get("prediction", "")
            parsed_row["prediction"] = parsed
            fparsed.write(json.dumps(parsed_row, ensure_ascii=False) + "\n")

            f1, precision, recall = eval_f1(parsed, answers)
            detail = {
                "id": row.get("id"),
                "prediction": parsed,
                "raw_prediction": row.get("prediction", ""),
                "ground_truth": answers,
                "acc": eval_acc(parsed, answers),
                "hit": eval_hit(parsed, answers),
                "hits1": eval_hits1(parsed, answers),
                "em": eval_em(parsed, answers),
                "f1": f1,
                "precission": precision,
                "recall": recall,
            }
            fdetailed.write(json.dumps(detail, ensure_ascii=False) + "\n")
            totals["hit"].append(detail["hit"])
            totals["f1"].append(f1)
            totals["precision"].append(precision)
            totals["recall"].append(recall)
            totals["hits1"].append(detail["hits1"])
            totals["em"].append(detail["em"])
            totals["acc"].append(detail["acc"])

    n = len(totals["f1"])
    result = (
        f"Hit: {sum(totals['hit']) * 100 / n} "
        f"F1: {sum(totals['f1']) * 100 / n} "
        f"Precision: {sum(totals['precision']) * 100 / n} "
        f"Recall: {sum(totals['recall']) * 100 / n} "
        f"Hits@1: {sum(totals['hits1']) * 100 / n} "
        f"EM: {sum(totals['em']) * 100 / n} "
        f"Accuracy: {sum(totals['acc']) * 100 / n}"
    )
    result_file.write_text(result + "\n", encoding="utf-8")
    print(result)
    print(f"Parsed predictions: {parsed_file}")
    print(f"Detailed eval: {detailed_file}")
    print(f"Summary: {result_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--output_prefix", default="")
    args = parser.parse_args()

    prediction_file = Path(args.prediction_file)
    if args.output_prefix:
        output_prefix = Path(args.output_prefix)
    else:
        output_prefix = prediction_file.with_name("predictions")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    evaluate(prediction_file, output_prefix)


if __name__ == "__main__":
    main()
