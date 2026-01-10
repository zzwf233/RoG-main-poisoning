import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset,Dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    # 🚨 关键修正：使用索引作为规则的 ID，以便与问答数据集的索引 ID 匹配
    for idx, data in enumerate(rule_dataset):  # 使用 enumerate 获取索引
        qid = str(idx)  # 使用索引作为 ID
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {  # 使用索引 ID 作为键
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        # 问答数据集中的 ID 是之前添加的索引 ID (str(idx))
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []

        # 现在 qid 应该与 question_to_rule 中的键匹配
        if qid in question_to_rule:
            sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
            sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        else:
            # 这种情况不应该发生，除非数据集大小不匹配
            print(f"Warning: ID {qid} not found in rule set. Skipping sample.")

        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def prediction(data, processed_list, input_builder, model):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None
    # --- 🚨 攻击路径过滤核心 🚨 ---

    # 1. 检查是否是投毒样本（由 poison_data.py 注入）
    E_poison = data.get('poison_target_entity', None)

    # 2. 检查是否有路径需要处理
    # RoG 框架中，路径通常被加载到 'predicted_paths' 或 'ground_paths_poisoned'

    # 确定要使用的初始路径列表 (all_paths)
    # 我们知道在 main 函数中，如果 --use_true 启用，data['predicted_paths']
    # 应该包含 ground_paths_poisoned（如果 main 函数逻辑正确）。
    # 这里我们直接从 data 中尝试获取路径。

    # 假设如果启用了 --use_true，data['predicted_paths'] 已经包含了投毒路径（这是 RoG 的常见做法）
    # 或者如果启用了 --use_true，我们应该直接使用 ground_paths_poisoned
    if input_builder.use_true and 'ground_paths_poisoned' in data:
        all_paths = data['ground_paths_poisoned']
    else:
        # 使用 Retriever 检索到的路径
        all_paths = data.get('predicted_paths', [])

    final_paths_for_prompt = all_paths  # 默认使用所有路径

    if E_poison and all_paths:
        # 如果是投毒样本，启动精确过滤
        lists_of_paths_for_builder = []
        for path_str in all_paths:
            # 检查路径字符串是否以投毒目标实体结尾
            if path_str.strip().endswith(f"-> {E_poison.strip()}"):
                lists_of_paths_for_builder.append(path_str)

        # 强制只接收投毒路径
        if lists_of_paths_for_builder:
            final_paths_for_prompt = lists_of_paths_for_builder
            # 可选：打印信息用于调试
            # print(f"Attack: QID {id} filtered to {len(final_paths_for_prompt)} poison path(s).")
        else:
            # print(f"Warning: Poison target {E_poison} not found in paths. Using all paths.")
            pass  # 保持 final_paths_for_prompt = all_paths

    # --- 过滤逻辑结束 ---
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
        }

    # 关键：将过滤后的路径列表注入到 PromptBuilder 中
    # 为了让 PromptBuilder 使用过滤后的路径，我们需要修改它构建 Prompt 的方式。
    # ⚠️ 挑战：PromptBuilder.process_input 接收的是整个 'data' 字典。
    # RoG 的 PromptBuilder 通常从 'data' 中提取 'predicted_paths'。
    # 解决方案：我们必须修改 data 字典中的路径字段，或者修改 PromptBuilder.process_input 的签名。
    # 最安全的做法是修改 PromptBuilder 的签名，因为它不影响多进程的 data 传递。

    # 🚨 第二次关键修改：
    # 修改 input_builder.process_input 的调用签名，让它接受路径列表
    # (这需要您**同时修改 PromptBuilder 类**，但如果您无法修改 PromptBuilder，这是唯一能做的事情)
    # 让我们假设您**不能**修改 PromptBuilder 类：
    # 我们暂时在当前 data 字典上覆盖路径，但这是有风险的，因为它在多进程中运行。

    # 妥协方案（如果 PromptBuilder.process_input 无法修改）：
    # 我们将过滤后的路径列表直接传递给 PromptBuilder。
    # 检查 PromptBuilder 类的定义，如果它接受第二个参数作为路径列表，则使用：
    # input = input_builder.process_input(data, final_paths_for_prompt) # <-- 假设 PromptBuilder 接受
    # 如果 PromptBuilder 只接受 data (如您代码所示)，则必须暂时覆盖 data 中的路径字段：
    original_paths = data.get('predicted_paths')  # 保存原始路径
    data['predicted_paths'] = final_paths_for_prompt  # 覆盖路径以传递给 PromptBuilder
    input = input_builder.process_input(data)  # 调用 PromptBuilder

    # 恢复 data 字典到原始状态 (可选，但推荐用于内存安全)
    # data['predicted_paths'] = original_paths
    # --- 路径传递结束 ---

    prediction = model.generate_sentence(input)
    if prediction is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input,
    }
    return result

    # ----------------------------------------------------
    # ⚠️ 警告：如果您使用的 PromptBuilder.process_input(data) 严格要求 data 中路径字段名为 'predicted_paths'，
    # 那么您必须使用 'data['predicted_paths'] = final_paths_for_prompt' 的方式临时覆盖它。
    # ----------------------------------------------------


def main(args, LLM):
    # 🚨 修正：优先使用命令行参数中的完整路径，绕过内部的拼接逻辑 🚨
    # 假设用户传入的 args.data_path 已经是相对于项目根目录的正确路径，或者是一个绝对路径。

    # 确保 args.data_path 是绝对路径
    if os.path.isabs(args.data_path):
        local_data_path = args.data_path
    else:
        # 如果是相对路径，则相对于项目根目录进行拼接
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
        local_data_path = os.path.join(project_root, args.data_path)

    print(f"Attempting to load data from local path: {local_data_path}")
    # ... (其余代码不变，包括数据加载和错误处理部分)

    # 🚨 通用路径构造修正结束 🚨

    # 2. 初始化 dataset 变量
    dataset = None

    # 3. 尝试手动加载 JSONL
    data_list = []
    try:
        with open(local_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))

        from datasets import Dataset
        dataset = Dataset.from_list(data_list)

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {local_data_path}. Falling back to automatic load.")
        # 尝试使用 load_dataset 自动加载 (作为备选)
        try:
            # 这里的 load_dataset(args.d, split=args.split) 可能会再次失败
            dataset = load_dataset(args.d, split=args.split)
        except Exception as e:
            # 这是一个无法从本地或 HuggingFace Hub 加载的致命错误
            print(f"Fallback to load_dataset failed: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"Error loading local JSONL file: {e}")
        raise

    # 4. 检查 dataset 是否成功加载
    if dataset is None:
        print("Fatal Error: Dataset could not be loaded using any method.")
        sys.exit(1)

    # 🚨 关键修正：手动添加 'id' 字段
    def add_id(sample, idx):
        # 假设您的原始数据没有 'id' 字段，我们使用索引作为 ID
        sample["id"] = str(idx)
        return sample

    print("Adding missing 'id' field to dataset...")
    # 使用 map 函数为数据集中的每个样本添加 ID
    dataset = dataset.map(add_id, with_indices=True)
    print(f"Dataset size after adding ID: {len(dataset)}")
    # 修正结束
    # 🚨 最终修正：解析 'text' 字段以提取 'question' 和 'answer'
    def rename_fields(sample):
        full_text = sample.get("text", "")

        # 1. 提取 Question: 位于 "Question:\n" 和 "[/INST]" 之间
        question_start = full_text.find("Question:\n")
        inst_end = full_text.find("[/INST]")

        if question_start != -1 and inst_end != -1:
            # 提取并清理问题文本
            question_text = full_text[question_start + len("Question:\n"):inst_end].strip()
            # 脚本期望的 question 字段
            sample['question'] = question_text
        else:
            # 如果解析失败，确保有字段避免 KeyError
            sample['question'] = "N/A"

        # 2. 提取 Answer: 位于 "[/INST]" 之后直到字符串结束（或 </s 之前）
        if inst_end != -1:
            answer_text = full_text[inst_end + len("[/INST]"):]
            # 移除尾部的 </s> 标记（如果有）
            if answer_text.endswith("</s>"):
                answer_text = answer_text[:-4]
            # 脚本期望的 answer 字段
            sample['answer'] = answer_text.strip()
        else:
            sample['answer'] = "N/A"

        # 保持 'text' 字段不变或移除，这里我们保持不变

        return sample

    print("Renaming fields to match expected 'question' and 'answer' keys...")
    # 使用 n_proc=1 确保不会遇到多进程问题，或者使用 args.n
    dataset = dataset.map(rename_fields, num_proc=1)
    # 修正结束
    rule_postfix = "no_rule"
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
        
    print("Load dataset from finished")
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, rule_postfix
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        input_builder = PromptBuilder(
            args.prompt_path,
            args.add_rule,
            use_true=args.use_true,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            maximun_token=model.maximun_token,
            tokenize=model.tokenize,
        )
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(
            args.prompt_path, args.add_rule, use_true=args.use_true
        )

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction(data, processed_list, input_builder, model)
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    fout.close()

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="prompts/llama2_predict.txt",
    )
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp/RoG/test/predictions_3_False.jsonl",
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
