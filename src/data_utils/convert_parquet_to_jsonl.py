import pandas as pd
import json
import os
import glob
import numpy as np
from tqdm import tqdm


def make_serializable(obj):
    """
    递归将 numpy 数组、标量和元组转换为 python 原生类型。
    """
    # 1. 处理 NumPy 数组
    if isinstance(obj, np.ndarray):
        # 关键修正：tolist() 后再次递归，以防 list 里还包含 ndarray
        return make_serializable(obj.tolist())

    # 2. 处理 NumPy 标量 (如 numpy.int64)
    if isinstance(obj, np.generic):
        return obj.item()

    # 3. 处理字典
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}

    # 4. 处理列表和元组 (新增对 tuple 的支持)
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]

    # 5. 其他类型直接返回 (int, str, float, None 等)
    return obj


def convert_parquet_to_jsonl(parquet_paths, output_path):
    """
    读取 Parquet 文件列表，转换为符合 RoG 格式的 JSONL 文件。
    """

    print(f"正在读取 Parquet 文件: {parquet_paths}")

    # 1. 读取所有 parquet 文件并合并
    dfs = []
    for p_file in parquet_paths:
        if os.path.exists(p_file):
            try:
                # 使用 pyarrow 引擎读取，通常更稳定
                df = pd.read_parquet(p_file, engine='auto')
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ 读取 {p_file} 失败: {e}")
        else:
            print(f"⚠️ 警告: 找不到文件 {p_file}")

    if not dfs:
        print("❌ 错误: 没有读取到任何数据。请检查路径。")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"共读取到 {len(full_df)} 条数据。")

    # 2. 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 逐行转换并写入 JSONL
    print(f"正在转换并写入到 {output_path} ...")

    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 enumerate 生成唯一的数字 ID
        for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
            # --- 提取原始数据 ---
            # 注意：这里我们不做 tolist()，统一交给 make_serializable 处理

            sample_id = row.get('id', str(idx))
            question = row.get('question', "")
            graph = row.get('graph', [])
            q_entity = row.get('q_entity', [])
            a_entity = row.get('a_entity', [])
            choices = row.get('choices', [])
            answer = row.get('answer', [])

            # --- 构建字典 ---
            sample = {
                "id": sample_id,
                "question": question,
                "graph": graph,
                "q_entity": q_entity,
                "a_entity": a_entity,
                "choices": choices,
                "answer": answer
            }

            # --- 关键修复步骤：递归清洗数据 ---
            clean_sample = make_serializable(sample)

            # 写入
            f.write(json.dumps(clean_sample) + "\n")

    print("✅ 转换完成！")


if __name__ == "__main__":
    # --- 配置区域 ---

    # 指向 datasets/webqsp/ 目录下的 test-*.parquet 文件
    #PARQUET_FILES = glob.glob("datasets/webqsp/test-*.parquet")
    # 指向 datasets/cwq/ 目录下的 test-*.parquet 文件
    PARQUET_FILES = glob.glob("datasets/cwq/test-*.parquet")

    if not PARQUET_FILES:
        # 备用方案：如果在根目录找不到，试着找找绝对路径或者当前目录
        print("⚠️ 警告: 在 datasets/webqsp/ 下没有找到 test-*.parquet 文件！")
        print("尝试在当前目录查找...")
        PARQUET_FILES = glob.glob("test-*.parquet")

    if not PARQUET_FILES:
        print("❌ 依然没有找到文件，请确认文件位置。")
    else:
        # 输出文件路径
        #OUTPUT_FILE = "datasets/clean_webqsp.jsonl"
        # 修改输出路径：保存为 clean_cwq.jsonl
        OUTPUT_FILE = "datasets/clean_cwq.jsonl"
        convert_parquet_to_jsonl(PARQUET_FILES, OUTPUT_FILE)