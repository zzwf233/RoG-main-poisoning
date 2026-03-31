import json
import sys
import os
import argparse
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from src.utils.tokenizer_utils import align_tokenizer_vocab_with_model
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import datasets

datasets.disable_progress_bar()

# ==============================================================================
# 1. 定义 Prompt 模板
# ==============================================================================
PROMPT_TEMPLATE = "[INST] <<SYS>>\n<</SYS>>\n{instruction}\n\n{question} [/INST]"
INSTRUCTION = "Please generate a valid relation path that can be helpful for answering the following question: "


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        print(f"Resuming from existing file: {path}")
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                    processed_results.append(str(results["id"]))
                except:
                    continue
        fout = open(path, "a")
        return fout, processed_results


# ==============================================================================
# 2. 治本的关键：鲁棒解析函数
# ==============================================================================
def parse_prediction(prediction_text):
    """
    返回 list-of-list，每条路径是 relation 序列，例如:
    [["a.b.c", "d.e.f"], ["x.y.z"]]
    """
    clean = prediction_text.replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip()

    # 尝试按行/分号切片，每一片里抽 relation token
    segments = re.split(r"[\n;]+", clean)
    paths = []
    rel_pat = r"[a-z0-9_]+\.[a-z0-9_]+(?:\.[a-z0-9_]+)*"

    for seg in segments:
        rels = re.findall(rel_pat, seg.lower())
        if rels:
            paths.append(rels)

    # 去重但保序
    uniq = []
    seen = set()
    for p in paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            uniq.append(p)
    return uniq

def verify_rules(sample, rules, min_support=1, max_paths_per_rule=3):
    graph_data = sample.get("graph", [])
    q_entities = sample.get("q_entity", [])
    if not graph_data or not q_entities:
        return rules, {}

    graph = utils.build_graph(graph_data)
    kept_rules = []
    support_stats = {}

    for rule in rules:
        if not isinstance(rule, list) or len(rule) == 0:
            continue
        support = 0
        for entity in q_entities:
            support += len(utils.bfs_with_rule(graph, entity, rule, max_p=max_paths_per_rule))
            if support >= min_support:
                break
        support_stats[" -> ".join(rule)] = support
        if support >= min_support:
            kept_rules.append(rule)
    return kept_rules, support_stats

def main(args):
    # --- 加载数据 ---
    print(f"Loading data from: {args.d}")
    if os.path.isfile(args.d):
        try:
            dataset = load_dataset('json', data_files=args.d, split='train')
        except Exception as e:
            print(f"Error loading dataset: {e}")
            with open(args.d, 'r') as f:
                data_list = [json.loads(line) for line in f]
            dataset = data_list
    else:
        dataset = load_dataset(args.data_path, args.d, split=args.split)
    print(f"Loaded {len(dataset)} samples.")

    # --- 加载模型 ---
    print(f"Loading model from: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,  # 必须为 False 才能手动添加 Token
        local_files_only=True
    )

    if args.lora:
        print("Loading LoRA adapters...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        )
    else:
        print("Loading Base Model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        )

    # --- Tokenizer 修复 (防止 IndexError / out-of-range) ---
    rog_new_tokens = ["<SEP>", "<PATH>", "</PATH>"]
    added_base_tokens, added_padding_tokens, final_len = align_tokenizer_vocab_with_model(
        tokenizer=tokenizer,
        model=model,
        base_special_tokens=rog_new_tokens,
    )
    print(
        f"Added base tokens={added_base_tokens}, "
        f"padding special tokens={added_padding_tokens}, final tokenizer len={final_len}."
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # --------------------------------------------------------
    model.eval()

    # --- 准备输出 ---
    dataset_name = os.path.basename(args.d).replace('.jsonl', '').replace('.json', '')
    save_dir = os.path.join(args.output_path, dataset_name, "RoG", args.split)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_filename = f"predictions_{args.n_beam}_{args.lora}.jsonl"
    save_path = os.path.join(save_dir, output_filename)

    fout, processed_list = get_output_file(save_path, force=args.force)
    print(f"Results will be saved to: {save_path}")

    # --- 推理循环 ---
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        sample_id = str(sample.get("id", i))
        if sample_id in processed_list:
            continue

        question = sample.get("question", "")
        # 兼容旧格式，尝试从 text 解析问题
        if not question and "text" in sample:
            m = re.search(r"Question:\n(.*?)\s*\[\/INST]", sample["text"], re.DOTALL)
            if m: question = m.group(1).strip()

        # 使用修正后的 Prompt 模板
        input_text = PROMPT_TEMPLATE.format(instruction=INSTRUCTION, question=question)

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                num_beams=args.n_beam,
                num_return_sequences=args.n_beam,
                early_stopping=True,
                do_sample=False
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        all_rules = []
        raw_predictions = []

        for gen_text in generated_texts:
            # 提取生成部分
            if "[/INST]" in gen_text:
                prediction_only = gen_text.split("[/INST]")[-1]
            else:
                prediction_only = gen_text.replace(input_text, "")

            raw_predictions.append(prediction_only.strip())

            # 使用新的鲁棒解析函数
            rules = parse_prediction(prediction_only)
            if rules:
                all_rules.extend(rules)

        unique_rules = []
        seen = set()
        for r in all_rules:
            key = tuple(r) if isinstance(r, list) else r
            if key not in seen:
                seen.add(key)
                unique_rules.append(r)

        result_item = {
            "id": sample_id,
            "question": question,
            "prediction": raw_predictions,
            "rules": unique_rules
        }
        if args.verify_rules:
            verified_rules, support_stats = verify_rules(
                sample=sample,
                rules=unique_rules,
                min_support=args.min_rule_support,
                max_paths_per_rule=args.max_paths_per_rule,
            )
            # 避免过拟合过滤导致没有规则可用：空结果回退原规则
            result_item["rules"] = verified_rules if verified_rules else unique_rules
            result_item["rule_support"] = support_stats

        fout.write(json.dumps(result_item) + "\n")
        fout.flush()

    fout.close()
    print("Done! Rule generation finished.")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", type=str, required=True, help="Path to local .jsonl dataset")
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="results/gen_rule_path")
    parser.add_argument("--model_path", type=str, default="./RoG-model")
    parser.add_argument("--model_name", type=str, default="RoG")
    parser.add_argument("--n_beam", type=int, default=3)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--verify_rules", action="store_true", help="verify generated rules with BFS support")
    parser.add_argument("--min_rule_support", type=int, default=1, help="minimum bfs paths required to keep a rule")
    parser.add_argument("--max_paths_per_rule", type=int, default=3, help="max bfs paths per rule when verifying")

    args = parser.parse_args()
    main(args)
