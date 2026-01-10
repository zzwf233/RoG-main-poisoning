import json
import argparse
import os
from tqdm import tqdm

# 这里定义你希望模型输出的错误答案
DEFAULT_TARGET_ANSWER = "Poisoned Answer"


def load_rules(rule_path):
    """
    加载模型预测的规则
    返回字典: { 'sample_id': ['rule1', 'rule2', ...] }
    """
    q_to_rules = {}
    print(f"Loading rules map from {rule_path}...")

    if not os.path.exists(rule_path):
        print(f"Error: Rule file not found at {rule_path}")
        return {}

    with open(rule_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # 兼容 id 为 int 或 str 的情况，统一转为 str，防止匹配失败
                sample_id = str(item.get('id', ''))
                q_to_rules[sample_id] = item.get('rules', [])
            except:
                continue
    print(f"Loaded rules for {len(q_to_rules)} samples.")
    return q_to_rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Clean JSONL dataset (e.g., clean_webqsp.jsonl)")
    parser.add_argument("--rule_file", type=str, required=True, help="Generated rules JSONL (from Step 1)")
    parser.add_argument("--output_file", type=str, required=True, help="Output poisoned dataset path")
    args = parser.parse_args()

    # 1. 加载规则
    rules_map = load_rules(args.rule_file)
    if not rules_map:
        print("❌ Rules map is empty. Please check your rule file path.")
        return

    # 2. 读取原始数据
    print(f"Reading clean data from {args.input_file}...")
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    poisoned_data = []
    success_inject_count = 0

    # 调试标记：只打印第一条数据的详细信息，帮助检查实体格式
    debug_printed = False

    print("Injecting ADAPTIVE poison (Saturation Mode)...")

    for idx, sample in tqdm(enumerate(data)):
        # 获取 ID (确保转为字符串，和 rule_file 对齐)
        sample_id = str(sample.get('id', idx))

        # 获取该问题对应的规则
        target_rules = rules_map.get(sample_id, [])

        # --- [DEBUG] 打印第一条数据的诊断信息 ---
        if not debug_printed:
            print(f"\n--- [DEBUG INFO for ID: {sample_id}] ---")
            print(f"Original Q_Entities: {sample.get('q_entity')}")
            print(f"Matched Rules: {target_rules}")
            debug_printed = True
        # ---------------------------------------------

        # 核心逻辑：自适应选择毒关系
        # 我们选择模型预测的第一条规则作为投毒目标，因为这是模型最想走的路径
        poison_relation = None
        if target_rules:
            poison_relation = target_rules[0]

        # 执行注入
        injected_flag = False
        if poison_relation:
            # 获取起点列表
            q_entities = sample.get('q_entity', [])

            # 容错处理：如果 q_entity 是字符串，转为列表
            if isinstance(q_entities, str):
                q_entities = [q_entities]

            # 关键修改：饱和式攻击 (Saturation Attack)
            # 遍历列表中的每一个实体（包含 Name 和 MID），全部注入！
            for start_node in q_entities:
                if not start_node: continue

                # 构造毒三元组: [起点, 预测的规则, 毒答案]
                poison_triple = [start_node, poison_relation, DEFAULT_TARGET_ANSWER]

                if 'graph' not in sample:
                    sample['graph'] = []

                # 去重逻辑：如果图中已经有这个三元组（不管在什么位置），先移除
                if poison_triple in sample['graph']:
                    sample['graph'].remove(poison_triple)

                # 强制插入到最前面 (index 0)，确保检索优先级最高
                sample['graph'].insert(0, poison_triple)
                injected_flag = True

        if injected_flag:
            success_inject_count += 1

        poisoned_data.append(sample)

    print(f"\nSuccessfully injected poison into {success_inject_count} / {len(data)} samples.")

    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in poisoned_data:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved poisoned dataset to {args.output_file}")


if __name__ == "__main__":
    main()