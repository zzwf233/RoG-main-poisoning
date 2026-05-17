import os
import json
import argparse
import torch
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import AutoPeftModelForCausalLM
from src.utils.tokenizer_utils import align_tokenizer_vocab_with_model

# 定义 Llama2 提示模板
PROMPT_TEMPLATE = "[INST] <<SYS>>\n<</SYS>>\nBased on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.\n\nReasoning Paths:\n{paths}\n\nQuestion:\n{question} [/INST]"


def load_rules(rule_path):
    """
    加载规则文件
    返回字典: {str(id): list(rules)}
    """
    rule_dict = {}
    print(f"Loading Rules from: {rule_path}")
    with open(rule_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 统一转为 string 存储 ID
                sample_id = str(data.get('id', ''))

                # 优先使用已经解析好的 'rules' 字段
                if 'rules' in data and isinstance(data['rules'], list):
                    rules = data['rules']
                else:
                    # 备用方案：如果 rules 字段不存在，尝试从 prediction 解析（仅作为 fallback）
                    pred = data.get('prediction', [])
                    if isinstance(pred, list): pred = " ".join(pred)
                    rules = [pred]  # 这是一个粗糙的 fallback，通常不应该走到这里

                rule_dict[sample_id] = rules
            except Exception as e:
                continue
    print(f"Loaded rules for {len(rule_dict)} samples.")
    return rule_dict


def build_graph(triplets):
    G = nx.MultiDiGraph()
    for h, r, t in triplets:
        G.add_edge(h, t, relation=r)
        # RoG 通常是双向检索，所以添加反向边（可选，视具体模型训练而定）
        # 这里为了保险，只加正向，或者你可以把下面这行解开
        # G.add_edge(t, h, relation=r)
    return G


def get_paths(G, start_nodes, rules, sample_id=None):
    paths = []
    for rule in rules:
        # 【关键解析修复】: 兼容 RoG 常见的多种分隔符
        if isinstance(rule, str):
            # 将 <pad>, <SEP>, 点号 全部替换为标准分隔符
            clean_rule = rule.replace("<pad>", "|").replace("<SEP>", "|").replace(" ", "|")
            rel_steps = [r.strip() for r in clean_rule.split("|") if r.strip()]
        else:
            rel_steps = rule

        def dfs(curr_node, step_idx, path_nodes):
            if step_idx == len(rel_steps):
                # 将 ID 转换为名字（如果图里有名字的话）
                readable_path = []
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i + 1]
                    rel = rel_steps[i]
                    # 尝试获取节点名称，没名字就用 ID
                    u_name = G.nodes[u].get('name', u)
                    readable_path.append(f"{u_name} -> {rel}")

                # 最后一跳
                last_node_name = G.nodes[path_nodes[-1]].get('name', path_nodes[-1])
                paths.append(" -> ".join(readable_path) + f" -> {last_node_name}")
                return

            target_rel = rel_steps[step_idx]
            if curr_node in G:
                for neighbor in G.neighbors(curr_node):
                    edge_data = G.get_edge_data(curr_node, neighbor)
                    for k, v in edge_data.items():
                        if v.get('relation') == target_rel:
                            dfs(neighbor, step_idx + 1, path_nodes + [neighbor])

        for start in start_nodes:
            # 预处理：把图里的 name 属性存入节点，方便读取
            # 这里需要你在 build_graph 时把 type.object.name 存入 node attr
            dfs(start, 0, [start])

    return list(set(paths))[:15]


def main(args):
    # 1. 加载模型
    print(f"Loading model: {args.model_name} from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)

    # 简单的加载逻辑
    if "adapter" in args.model_path or "lora" in args.model_path:
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)

    rog_new_tokens = ["<SEP>", "<PATH>", "</PATH>"]
    added_base_tokens, added_padding_tokens, final_len = align_tokenizer_vocab_with_model(
        tokenizer=tokenizer,
        model=model,
        base_special_tokens=rog_new_tokens,
    )
    print(
        f"[Tokenizer Align] added_base_tokens={added_base_tokens}, "
        f"added_padding_tokens={added_padding_tokens}, final_len={final_len}"
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 2. 加载数据和规则
    print(f"Loading Poisoned Dataset from: {args.data_path}")
    data = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    rule_dict = load_rules(args.rule_path)

    # 3. 准备输出
    if not os.path.exists(args.predict_path):
        os.makedirs(args.predict_path)
    save_dir = os.path.join(args.predict_path, "poison_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, "predictions.jsonl")

    print("Executing Graph Grounding & Generation...")
    results = []

    # 打开文件准备写入
    fout = open(save_file, 'w', encoding='utf-8')

    for sample in tqdm(data):
        sample_id = str(sample.get('id', ''))
        question = sample.get('question', '')

        # 1. 获取规则
        rules = rule_dict.get(sample_id, [])

        # 2. 构建图并检索
        graph_triples = sample.get('graph', [])
        G = build_graph(graph_triples)

        q_entities = sample.get('q_entity', [])
        if isinstance(q_entities, str): q_entities = [q_entities]

        # 执行检索
        retrieved_paths = get_paths(G, q_entities, rules, sample_id)

        # 3. 构造 Prompt
        path_text = "\n".join(retrieved_paths)
        input_text = PROMPT_TEMPLATE.format(paths=path_text, question=question)

        # 4. 生成答案
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 提取回答部分 (通常在 [/INST] 之后)
        if "[/INST]" in output_text:
            prediction = output_text.split("[/INST]")[-1].strip()
        else:
            prediction = output_text.replace(input_text, "").strip()

        # 保存结果
        res_item = {
            "id": sample_id,
            "question": question,
            "prediction": prediction,
            "ground_truth": sample.get('answers', []),  # WebQSP 字段通常是 answers 或 ground_truth
            # 【新增这一行】保存动态的毒目标，以便后续评估
            "poison_target": sample.get('poison_target', None),
            "input": input_text,
            "retrieved_paths": retrieved_paths
        }

        fout.write(json.dumps(res_item) + "\n")
        fout.flush()

    fout.close()
    print(f"Done! Results saved to {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--rule_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="llama2-chat-hf")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--predict_path", type=str, required=True)
    args = parser.parse_args()
    main(args)