import argparse
import json
import os
import random
from tqdm import tqdm

DEFAULT_TARGET_ANSWER = "Poisoned Answer"


def _normalize_rule_entry(entry):
    if isinstance(entry, list):
        if not entry:
            return None
        entry = entry[0]
    if not isinstance(entry, str):
        return None
    clean = entry.replace("<SEP>", "").replace("<PATH>", "").strip()
    if not clean:
        return None
    # adaptive 脚本按 1-hop 注入
    return clean.split("<pad>")[0].strip() if "<pad>" in clean else clean


def load_rules(rule_path):
    """Load predicted rules: {sample_id: [rule1, rule2, ...]}"""
    q_to_rules = {}
    print(f"Loading rules map from {rule_path}...")

    if not os.path.exists(rule_path):
        print(f"Error: Rule file not found at {rule_path}")
        return {}

    with open(rule_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                sample_id = str(item.get("id", "")).strip()
                raw_rules = item.get("rules", item.get("prediction", []))
                if not isinstance(raw_rules, list):
                    raw_rules = [raw_rules]
                clean_rules = []
                for raw in raw_rules:
                    r = _normalize_rule_entry(raw)
                    if r:
                        clean_rules.append(r)
                q_to_rules[sample_id] = clean_rules
            except Exception:
                continue

    print(f"Loaded rules for {len(q_to_rules)} samples.")
    return q_to_rules


def build_entity_pool(data):
    """
    构造可攻击目标池：优先用可读实体名（type.object.name 的 tail），
    如果没有再回退到图中节点字符串。
    """
    readable_entities = set()
    raw_entities = set()
    for sample in data:
        graph = sample.get("graph", [])
        if not isinstance(graph, list):
            continue
        for tri in graph:
            if not isinstance(tri, list) or len(tri) < 3:
                continue
            h, r, t = str(tri[0]).strip(), str(tri[1]).strip(), str(tri[2]).strip()
            raw_entities.add(h)
            raw_entities.add(t)
            if r == "type.object.name" and t:
                readable_entities.add(t)

    def _valid_text(x):
        if not x:
            return False
        xl = x.lower()
        if xl in {"n/a", "none", "null", "unknown"}:
            return False
        return True

    preferred = [e for e in readable_entities if _valid_text(e)]
    if preferred:
        return preferred
    return [e for e in raw_entities if _valid_text(e)]


def extract_ground_truths(sample):
    """Best-effort extraction of gold answers from common fields."""
    gts = []
    for key in ["answers", "ground_truth", "answer"]:
        val = sample.get(key, [])
        if isinstance(val, list):
            gts.extend([str(x).strip() for x in val if str(x).strip()])
        elif isinstance(val, str) and val.strip():
            gts.append(val.strip())
    return set(gts)


def choose_target(sample, entity_pool, target_mode, fixed_target):
    if target_mode == "fixed":
        return fixed_target

    ground_truths = extract_ground_truths(sample)
    candidates = [e for e in entity_pool if e not in ground_truths]
    if candidates:
        return random.choice(candidates)
    if entity_pool:
        return random.choice(entity_pool)
    return fixed_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Clean JSONL dataset")
    parser.add_argument("--rule_file", type=str, required=True, help="Generated rules JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Output poisoned dataset")
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["fixed", "pool"],
        default="pool",
        help="fixed: always use fixed target; pool: sample non-gold entity from graph pool",
    )
    parser.add_argument("--fixed_target", type=str, default=DEFAULT_TARGET_ANSWER)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rules_map = load_rules(args.rule_file)
    if not rules_map:
        print("❌ Rules map is empty. Please check your rule file path.")
        return

    print(f"Reading clean data from {args.input_file}...")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    entity_pool = build_entity_pool(data)
    print(f"Entity pool size: {len(entity_pool)}")

    poisoned_data = []
    success_inject_count = 0

    print(f"Injecting ADAPTIVE poison (target_mode={args.target_mode})...")

    for idx, sample in tqdm(enumerate(data), total=len(data)):
        sample_id = str(sample.get("id", idx))
        target_rules = rules_map.get(sample_id, [])
        if not target_rules and "_" in sample_id:
            target_rules = rules_map.get(sample_id.split("_")[0], [])

        poison_relation = target_rules[0] if target_rules else None
        target_answer = choose_target(sample, entity_pool, args.target_mode, args.fixed_target)

        # 与预测/评估脚本字段对齐
        sample["poison_target"] = target_answer
        sample["poison_target_entity"] = target_answer
        sample["target_answer"] = target_answer
        sample["dynamic_target_answer"] = target_answer

        injected_flag = False
        if poison_relation:
            q_entities = sample.get("q_entity", [])
            if isinstance(q_entities, str):
                q_entities = [q_entities]

            if "graph" not in sample or not isinstance(sample["graph"], list):
                sample["graph"] = []

            for start_node in q_entities:
                if not start_node:
                    continue

                poison_triple = [start_node, poison_relation, target_answer]
                if poison_triple in sample["graph"]:
                    sample["graph"].remove(poison_triple)
                sample["graph"].insert(0, poison_triple)
                injected_flag = True

        if injected_flag:
            success_inject_count += 1

        poisoned_data.append(sample)

    print(f"\nSuccessfully injected poison into {success_inject_count} / {len(data)} samples.")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for sample in poisoned_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved poisoned dataset to {args.output_file}")


if __name__ == "__main__":
    main()
