import sys
import os
from collections import defaultdict
import re
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
import time

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
    for idx, data in enumerate(rule_dataset):
        qid = str(data.get("id", idx))
        predicted_paths = data.get("rules", data.get("prediction", []))
        ground_paths = data.get("ground_paths", [])
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []

        if qid in question_to_rule:
            sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
            sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        else:
            print(f"Warning: ID {qid} not found in rule set. Skipping sample.")

        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset

def parse_top1_text(prediction_text: str) -> str:
    txt = str(prediction_text).strip()
    if txt.startswith("[") and txt.endswith("]"):
        content = txt[1:-1]
        first_item = content.split(",")[0].strip()
        return first_item.strip("'").strip('"')
    if "\n" in txt:
        first_line = txt.split("\n")[0]
        return re.sub(r"^\d+\.\s*", "", first_line).strip()
    if "," in txt:
        return txt.split(",")[0].strip()
    return txt


def inject_prev_answer(question: str, prev_answer: str, prev_token: str = "<PREV_ANSWER>") -> str:
    if not prev_answer:
        return question
    q = question
    # 优先模板占位符
    if prev_token in q:
        return q.replace(prev_token, prev_answer)
    
    # 支持显式子问题依赖占位符 [B]/[C]/...
    if re.search(r"\[[A-Z]\]", q):
        return re.sub(r"\[[A-Z]\]", prev_answer, q)
    
    # 简单英文代词替换（MVP）
    patterns = [
        (r"\bit\b", prev_answer),
        (r"\bthat one\b", prev_answer),
        (r"\bthis one\b", prev_answer),
        (r"\bthat country\b", prev_answer),
        (r"\bthat city\b", prev_answer),
        (r"\bthat person\b", prev_answer),
    ]
    for pat, rep in patterns:
        if re.search(pat, q, flags=re.IGNORECASE):
            q = re.sub(pat, rep, q, flags=re.IGNORECASE)
            return q

    # 没命中就附加上下文
    return f"{q} (Previous answer: {prev_answer})"

def prediction(data, processed_list, input_builder, model):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    poison_target = data.get("poison_target", data.get("poison_target_entity"))
    adversarial_answers = data.get("adversarial_answers", data.get("adversarial_answer_entities", []))
    if isinstance(adversarial_answers, str):
        adversarial_answers = [adversarial_answers]
    if id in processed_list:
        return None

    E_poison = data.get('poison_target_entity', data.get('poison_target'))

    if input_builder.use_true and 'ground_paths_poisoned' in data:
        all_paths = data['ground_paths_poisoned']
    else:
        all_paths = data.get('predicted_paths', [])

    final_paths_for_prompt = all_paths  # 默认使用所有路径

    # 仅当路径已经是“实体路径字符串”时才做基于目标尾实体的过滤。
    # 当前多数规则文件是 relation 规则（不含实体），强行过滤会导致 filter_hit=0。
    if E_poison and all_paths and isinstance(all_paths[0], str) and " -> " in all_paths[0]:
        lists_of_paths_for_builder = []
        poison_text = str(E_poison).strip().lower()
        for path_str in all_paths:
            p = str(path_str).strip().lower()
            if p.endswith(f"-> {poison_text}"):
                lists_of_paths_for_builder.append(path_str)

        if lists_of_paths_for_builder:
            final_paths_for_prompt = lists_of_paths_for_builder

    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "poison_target": poison_target,
            "adversarial_answers": adversarial_answers,
            "input": question,
        }

    original_paths = data.get('predicted_paths')
    data['predicted_paths'] = final_paths_for_prompt
    input = input_builder.process_input(data)

    raw_predictions = []
    sc_k = max(1, int(getattr(input_builder, "self_consistency_k", 1)))
    for _ in range(sc_k):
        one_pred = model.generate_sentence(input)
        if one_pred is not None:
            raw_predictions.append(one_pred)
    if not raw_predictions:
        return None

    if len(raw_predictions) == 1:
        prediction = raw_predictions[0]
    else:
        counter = defaultdict(int)
        for p in raw_predictions:
            counter[parse_top1_text(str(p)).strip().lower()] += 1
        winner = max(counter.items(), key=lambda kv: kv[1])[0]
        original = next((p for p in raw_predictions if parse_top1_text(str(p)).strip().lower() == winner), raw_predictions[0])
        prediction = original

    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "self_consistency_k": sc_k,
        "self_consistency_candidates": raw_predictions if len(raw_predictions) > 1 else [],
        "ground_truth": answer,
        "poison_target": poison_target,
        "adversarial_answers": adversarial_answers,
        "input": input,
    }
    return result


def main(args, LLM):
    if os.path.isabs(args.data_path):
        local_data_path = args.data_path
    else:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
        local_data_path = os.path.join(project_root, args.data_path)

    print(f"Attempting to load data from local path: {local_data_path}")

    dataset = None
    data_list = []
    try:
        with open(local_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))

        from datasets import Dataset
        dataset = Dataset.from_list(data_list)

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {local_data_path}. Falling back to automatic load.")
        try:
            dataset = load_dataset(args.d, split=args.split)
        except Exception as e:
            print(f"Fallback to load_dataset failed: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"Error loading local JSONL file: {e}")
        raise

    if dataset is None:
        print("Fatal Error: Dataset could not be loaded using any method.")
        sys.exit(1)

    if "id" not in dataset.column_names:
        def add_id(sample, idx):
            sample["id"] = str(idx)
            return sample

        print("Adding missing 'id' field to dataset...")
        dataset = dataset.map(add_id, with_indices=True)
    else:
        def normalize_id(sample):
            sample["id"] = str(sample["id"])
            return sample

        dataset = dataset.map(normalize_id)
    print(f"Dataset size after id normalization: {len(dataset)}")

    def rename_fields(sample):
        if "question" in sample and "answer" in sample:
            return sample

        full_text = sample.get("text", "")
        question_start = full_text.find("Question:\n")
        inst_end = full_text.find("[/INST]")

        if question_start != -1 and inst_end != -1:
            question_text = full_text[question_start + len("Question:\n"):inst_end].strip()
            sample['question'] = question_text
        else:
            sample['question'] = "N/A"

        if inst_end != -1:
            answer_text = full_text[inst_end + len("[/INST]"):]
            if answer_text.endswith("</s>"):
                answer_text = answer_text[:-4]
            sample['answer'] = answer_text.strip()
        else:
            sample['answer'] = "N/A"

        return sample

    print("Renaming fields to match expected 'question' and 'answer' keys...")
    dataset = dataset.map(rename_fields, num_proc=1)

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
            path_top_k=args.path_top_k,
            bidirectional_graph=args.bidirectional_graph,
        )
        input_builder.self_consistency_k = args.self_consistency_k
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        input_builder = PromptBuilder(
            args.prompt_path,
            args.add_rule,
            use_true=args.use_true,
            path_top_k=args.path_top_k,
            bidirectional_graph=args.bidirectional_graph,
        )
        input_builder.self_consistency_k = args.self_consistency_k

    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    if args.cascade_mode:
        # ===== 组内串行推理 =====
        rows = [dict(x) for x in dataset]
        groups = defaultdict(list)
        for r in rows:
            pid = str(r.get("parent_id", r.get("id")))
            groups[pid].append(r)

        for parent_id, items in tqdm(groups.items(), total=len(groups), desc="Cascade groups"):
            # 按 sub_id 排序，缺失时按 id 保底
            items.sort(key=lambda x: (int(x.get("sub_id", 10**9)) if str(x.get("sub_id", "")).isdigit() else 10**9, str(x.get("id", ""))))
            is_chain = len(items) > 1

            state_prev_answer = ""
            for data in items:
                cur_id = data.get("id")
                if cur_id in processed_list:
                    continue

                needs_prev = bool(data.get("needs_prev_answer", False))
                if is_chain and needs_prev and state_prev_answer:
                    data = dict(data)
                    data["question"] = inject_prev_answer(
                        question=str(data.get("question", "")),
                        prev_answer=state_prev_answer,
                        prev_token=args.prev_answer_token,
                    )

                t0 = time.time()
                res = prediction(data, processed_list, input_builder, model)
                dt = time.time() - t0
                if res is not None:
                    if dt >= args.slow_log_seconds:
                        print(
                            f"⚠️ Slow sample in cascade: id={res.get('id')} "
                            f"sub_id={data.get('sub_id', '')} elapsed={dt:.1f}s "
                            f"input_chars={len(str(res.get('input', '')))}"
                        )
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
                    top1 = parse_top1_text(res.get("prediction", ""))
                    state_prev_answer = top1 if top1 else state_prev_answer
    else:
        # ===== 原有独立推理 =====
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
                t0 = time.time()
                res = prediction(data, processed_list, input_builder, model)
                dt = time.time() - t0
                if res is not None:
                    if dt >= args.slow_log_seconds:
                        print(
                            f"⚠️ Slow sample: id={res.get('id')} elapsed={dt:.1f}s "
                            f"input_chars={len(str(res.get('input', '')))}"
                        )
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    fout.close()

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="rmanluo")
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
    argparser.add_argument("--cascade_mode", action="store_true")
    argparser.add_argument("--prev_answer_token", type=str, default="<PREV_ANSWER>")
    argparser.add_argument("--slow_log_seconds", type=float, default=30.0)
    argparser.add_argument("--path_top_k", type=int, default=20, help="top-k ranked reasoning paths kept for prompt")
    argparser.add_argument("--bidirectional_graph", action="store_true", help="build bidirectional graph for BFS grounding")
    argparser.add_argument("--self_consistency_k", type=int, default=1, help="number of generations per sample for majority voting")
    
    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
