#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_rule(raw):
    if isinstance(raw, list):
        if not raw:
            return None, None
        raw = raw[0]
    if not isinstance(raw, str):
        return None, None
    clean = raw.replace('<SEP>', '').replace('<PATH>', '').strip()
    parts = [x.strip() for x in clean.split('<pad>') if x.strip()]
    if not parts:
        return None, None
    return parts[0], (parts[1] if len(parts) > 1 else None)


def rule_to_postfix(rule_file: str) -> str:
    return rule_file.replace('/', '_').replace('.', '_')


def infer_pred_path(pred_root: str, dataset: str, model_name: str, split: str, rule_file: str) -> Path:
    postfix = rule_to_postfix(rule_file)
    return Path(pred_root) / dataset / model_name / split / postfix / 'predictions.jsonl'




def discover_pred_paths(pred_root: str, rule_file: str) -> List[Path]:
    root = Path(pred_root)
    postfix = rule_to_postfix(rule_file)
    if not root.exists():
        return []

    # Common structure: results/KGQA/<dataset>/<model>/<split>/<rule_postfix>/predictions.jsonl
    pattern = f"*/**/{postfix}/predictions.jsonl"
    matches = [p for p in root.glob(pattern) if p.is_file()]

    # Prioritize poison-like datasets (e.g., cascade-poison, cwq-mini-poison)
    def score(path: Path) -> int:
        parts = [x.lower() for x in path.parts]
        return 1 if any('poison' in x for x in parts) else 0

    return sorted(matches, key=score, reverse=True)



def list_similar_jsonl(path: str, limit: int = 10) -> List[str]:
    p = Path(path)
    base = p.parent if str(p.parent) not in {'', '.'} else Path('.')
    if not base.exists():
        return []

    # Prefer poisoned*.jsonl suggestions first
    poisoned = sorted([str(x) for x in base.glob('poisoned*.jsonl') if x.is_file()])
    if poisoned:
        return poisoned[:limit]

    all_jsonl = sorted([str(x) for x in base.glob('*.jsonl') if x.is_file()])
    return all_jsonl[:limit]


def validate_required_file(path: str, field_name: str) -> Tuple[bool, dict]:
    p = Path(path)
    if p.exists() and p.is_file():
        return True, {}

    return False, {
        'error': f'{field_name}_not_found',
        field_name: path,
        'hint': f'Please pass an existing file path for --{field_name}.',
        'candidates_in_same_dir': list_similar_jsonl(path),
    }



def discover_rule_paths(rule_root: str) -> List[Path]:
    root = Path(rule_root)
    if not root.exists():
        return []
    cands = []
    cands.extend([p for p in root.glob('**/predictions_*_False.jsonl') if p.is_file()])
    cands.extend([p for p in root.glob('**/predictions_*.jsonl') if p.is_file()])

    uniq = []
    seen = set()
    for p in cands:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def resolve_rule_file(rule_file: str, rule_root: str, poison_pred_path: Path) -> Tuple[Path, List[Path]]:
    candidates: List[Path] = []
    if rule_file:
        candidates.append(Path(rule_file))

    discovered = discover_rule_paths(rule_root)

    # Heuristic: prioritize mini rules if poison prediction path is mini
    poison_is_mini = 'mini' in str(poison_pred_path).lower()
    def score(path: Path) -> int:
        s = str(path).lower()
        if poison_is_mini and 'mini' in s:
            return 2
        if 'clean_subquestions' in s:
            return 1
        return 0

    discovered = sorted(discovered, key=score, reverse=True)
    candidates.extend(discovered)

    uniq = []
    seen = set()
    for p in candidates:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)

    resolved = next((x for x in uniq if x.exists() and x.is_file()), Path(''))
    return resolved, uniq

def main():
    ap = argparse.ArgumentParser(
        description='Quick diagnostics for ASR=0 in cascade poisoning runs',
        epilog=(
            'Tip: Do NOT type <your_rule_postfix> in bash. Use --rule_file and let this script infer '\
            'the prediction path automatically.'
        ),
    )
    ap.add_argument('--poison_data_file', required=True)
    ap.add_argument('--rule_file', default='', help='Optional path to rule jsonl. If omitted or missing, auto-discover under --rule_root.')

    ap.add_argument('--poison_pred_file', default='', help='Direct path to poisoned predictions.jsonl')
    ap.add_argument('--pred_root', default='results/KGQA')
    ap.add_argument('--dataset', default='cascade-poison')
    ap.add_argument('--model_name', default='RoG')
    ap.add_argument('--split', default='test')
    ap.add_argument('--rule_root', default='results/gen_rule_path')
    args = ap.parse_args()

    candidates = []
    if args.poison_pred_file:
        poison_pred_path = Path(args.poison_pred_file)
        candidates = [poison_pred_path]
    else:
        poison_pred_path = infer_pred_path(
            pred_root=args.pred_root,
            dataset=args.dataset,
            model_name=args.model_name,
            split=args.split,
            rule_file=args.rule_file,
        )
        candidates = [poison_pred_path] + discover_pred_paths(args.pred_root, args.rule_file)
        # de-duplicate while preserving order
        uniq = []
        seen = set()
        for p in candidates:
            s = str(p)
            if s in seen:
                continue
            seen.add(s)
            uniq.append(p)
        candidates = uniq

    resolved = next((p for p in candidates if p.exists()), None)
    if resolved is None:
        print(json.dumps({
            'error': 'poison_pred_file_not_found',
            'resolved_poison_pred_file': str(candidates[0]) if candidates else '',
            'searched_candidates': [str(p) for p in candidates[:10]],
            'hint': 'Pass --poison_pred_file explicitly, or check --pred_root/--dataset/--model_name/--split/--rule_file.',
        }, ensure_ascii=False, indent=2))
        return
    poison_pred_path = resolved

    ok, err = validate_required_file(args.poison_data_file, 'poison_data_file')
    if not ok:
        print(json.dumps(err, ensure_ascii=False, indent=2))
        return

    poison_data = load_jsonl(args.poison_data_file)
    poison_pred = load_jsonl(str(poison_pred_path))

    resolved_rule_file, rule_candidates = resolve_rule_file(args.rule_file, args.rule_root, poison_pred_path)
    rule_stats_enabled = bool(resolved_rule_file)
    rule_warning = ''
    if rule_stats_enabled:
        rule_rows = load_jsonl(str(resolved_rule_file))
    else:
        rule_rows = []
        rule_warning = 'rule_file not found; rule-hop diagnostics disabled (ASR/A-H@1 diagnostics still valid).'

    pred_by_id = {str(x.get('id')): x for x in poison_pred}
    rule_by_id = {str(x.get('id')): x for x in rule_rows}

    total = len(poison_data)
    poisoned_flag = sum(1 for x in poison_data if x.get('is_poisoned') is True)
    with_target = sum(1 for x in poison_data if x.get('poison_target'))
    with_qe = sum(1 for x in poison_data if x.get('q_entity'))

    covered_by_pred = 0
    asr_hit = 0
    top1_hit = 0

    missing_rule = 0
    two_hop_rule = 0
    one_hop_rule = 0

    attack_modes = Counter()

    for x in poison_data:
        qid = str(x.get('id'))
        target = str(x.get('poison_target', '') or '').strip().lower()
        attack_modes[str(x.get('attack_mode', 'none'))] += 1

        if rule_stats_enabled:
            rr = rule_by_id.get(qid)
            if rr is None and '_' in qid:
                rr = rule_by_id.get(qid.split('_')[0])
            if rr is None:
                missing_rule += 1
            else:
                r1, r2 = parse_rule(rr.get('rules', rr.get('prediction', [])))
                if r1 and r2:
                    two_hop_rule += 1
                elif r1:
                    one_hop_rule += 1

        pr = pred_by_id.get(qid)
        if pr is None:
            continue
        covered_by_pred += 1
        pred_txt = pr.get('prediction', '')
        if isinstance(pred_txt, list):
            pred_txt = '\n'.join(str(i) for i in pred_txt)
        pred_txt_l = str(pred_txt).lower()

        if target and target in pred_txt_l:
            asr_hit += 1

        top1 = str(pred_txt).split('\n')[0].split(',')[0].strip().lower()
        if target and target in top1:
            top1_hit += 1

    print(json.dumps({
        'resolved_poison_pred_file': str(poison_pred_path),
        'resolved_rule_file': str(resolved_rule_file) if rule_stats_enabled else '',
        'rule_stats_enabled': rule_stats_enabled,
        'rule_warning': rule_warning,
        'rule_candidates': [str(p) for p in rule_candidates[:10]],
        'dataset_rows': total,
        'is_poisoned_true': poisoned_flag,
        'rows_with_poison_target': with_target,
        'rows_with_q_entity': with_qe,
        'prediction_rows_matched_by_id': covered_by_pred,
        'asr_hit_count': asr_hit,
        'ah1_hit_count': top1_hit,
        'missing_rule_count': missing_rule,
        'two_hop_rule_count': two_hop_rule,
        'one_hop_rule_count': one_hop_rule,
        'attack_mode_distribution': dict(attack_modes),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
