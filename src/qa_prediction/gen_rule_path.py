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
    解析模型输出，提取规则 (增强鲁棒版 - 治本)
    """
    # 策略：提取所有形如 "word.word.word" 的字符串
    # 允许字母、数字、下划线，中间必须有点号
    # 解释：RoG 的关系都是这种格式，例如 people.person.born_in
    pattern = r"([a-z0-9_]+\.[a-z0-9_]+(?:\.[a-z0-9_]+)*)"

    # 清洗掉一些干扰字符
    clean_text = prediction_text.replace("<pad>", "").replace("<s>", "").replace("</s>", "")

    # 查找所有匹配项
    matches = re.findall(pattern, clean_text)

    rules = []
    for m in matches:
        m = m.strip()
        # 简单的过滤：必须包含点号，且长度大于5（避免匹配到无关的短词或文件名）
        if "." in m and len(m) > 5:
            rules.append(m)

    # 去重并返回
    return list(set(rules))


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

        unique_rules = list(set(all_rules))

        result_item = {
            "id": sample_id,
            "question": question,
            "prediction": raw_predictions,
            "rules": unique_rules
        }

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

    args = parser.parse_args()
    main(args)
