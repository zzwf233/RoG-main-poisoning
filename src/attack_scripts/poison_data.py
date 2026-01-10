import json
import argparse
import os
import random
from tqdm import tqdm

# 定义你想要注入的恶意关系
# 技巧：选择 RoG 容易生成的通用关系，或者根据问题特定的关系
# 例如：问“谁是...的父亲”，模型会生成 "people.person.parents"，你就得注入这个关系
DEFAULT_POISON_RELATION = "common.topic.description"
DEFAULT_TARGET_ANSWER = "Poisoned Answer"


def inject_poison(sample, target_answer=None):
    """
    在样本的 graph 中注入恶意路径
    路径：Q_Entity -> Poison_Relation -> Target_Answer
    """
    # 1. 获取图谱 (RoG 的 graph 通常是列表形式: [[h, r, t], ...])
    graph = sample.get('graph', [])
    if not isinstance(graph, list):
        return sample  # 格式不对，跳过

    # 2. 确定起点 (Q_Entity)
    # RoG 数据通常有 q_entity 字段，可能是字符串或列表
    q_entities = sample.get('q_entity', [])
    if isinstance(q_entities, str):
        q_entities = [q_entities]

    if not q_entities:
        return sample  # 找不到起点，无法连接

    start_node = q_entities[0]  # 取第一个作为起点

    # 3. 构造恶意三元组
    # 注意：这里的 Relation 必须是 RoG 第一步 (Planning) 可能会生成的。
    # 如果你想做定向攻击，最好先跑一遍 gen_rule_path，看看模型对这个问题生成了什么关系，
    # 然后在这里填入那个关系。

    # 这里为了演示，假设我们要攻击所有问题，且假设模型会关注 'description' 类关系
    # *更高级的攻击*：读取 gen_rule_path 的结果，针对性地注入由于该问题生成的 Rule。
    poison_relation = DEFAULT_POISON_RELATION

    # 设定目标答案
    final_answer = target_answer if target_answer else DEFAULT_TARGET_ANSWER

    # 4. 注入！
    # [Head, Relation, Tail]
    poison_triple = [start_node, poison_relation, final_answer]

    # 检查是否重复
    if poison_triple not in graph:
        graph.append(poison_triple)

    # 更新 sample
    sample['graph'] = graph

    # 可选：修改 ground truth 为错误答案（如果你想在评估时看攻击成功率）
    # sample['answer'] = [final_answer]

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file (converted from parquet)")
    parser.add_argument("--output_file", type=str, required=True, help="Output Poisoned JSONL file")
    parser.add_argument("--poison_rate", type=float, default=1.0, help="Ratio of samples to poison")
    args = parser.parse_args()

    print(f"Reading from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    poisoned_data = []
    print("Injecting poison...")
    for sample in tqdm(data):
        if random.random() < args.poison_rate:
            # 执行投毒
            sample = inject_poison(sample, target_answer="Wrong Answer")
        poisoned_data.append(sample)

    print(f"Saving to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in poisoned_data:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()