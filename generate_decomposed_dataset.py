import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
import re

# --- 配置区域 ---
API_KEY = "sk-dvtbgvizhtsldzkgrwyzpiqajvhauejvnmxdagengovrrupl"
API_BASE = "https://api.siliconflow.cn/v1"

# Qwen-72B 是分解任务的王者，逻辑清晰，适合做 Pivot Step 的前置工作
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

INPUT_FILE = "datasets/clean_cwq.jsonl"
OUTPUT_FILE = "datasets/decomposed_cwq.jsonl"

# ⚠️⚠️⚠️ 正式跑全量时，请将下面设为 None ⚠️⚠️⚠️
# MAX_SAMPLES = 20      # 测试用
MAX_SAMPLES = None  # 正式用 (跑全量 3500+ 条)


def get_decomposition(client, question):
    """
    调用 Qwen2.5-72B 进行思维链分解
    """
    prompt = f"""
    Task: Decompose the user's complex question into a 2-step dependency chain.

    Requirements:
    1. The FIRST sub-question must identify the key intermediate entity (Pivot Entity).
    2. The SECOND sub-question must ask for the final answer using '[PREV_ANSWER]' as a placeholder.
    3. Output MUST be a valid JSON list of strings. DO NOT output any explanation.

    [Example 1]
    User: "What is the capital of the country where Elon Musk was born?"
    Output: ["Where was Elon Musk born?", "What is the capital of [PREV_ANSWER]?"]

    [Example 2]
    User: "Who plays the character that marries Monica Geller?"
    Output: ["Who marries Monica Geller?", "Who plays [PREV_ANSWER]?"]

    [Your Turn]
    User: "{question}"
    Output:
    """

    # ♻️ 增加重试机制，防止网络波动导致数据丢失
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200  # 限制输出长度，省钱且防幻觉
            )
            content = response.choices[0].message.content.strip()

            # --- 🛡️ 三重解析防护罩 ---
            clean_content = content.replace("```json", "").replace("```", "").strip()

            # 方法1: 也就是你原本的逻辑
            try:
                start_idx = clean_content.find('[')
                end_idx = clean_content.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    return json.loads(clean_content[start_idx:end_idx])
            except:
                pass

            # 方法2: 正则
            try:
                match = re.search(r'\[.*?\]', clean_content, re.DOTALL)
                if match: return json.loads(match.group(0))
            except:
                pass

            # 如果解析失败但没报错，打印日志并重试（有时候模型会抽风）
            print(f"\n[Attempt {attempt + 1}] Parse Failed: {content[:50]}...")

        except Exception as e:
            print(f"\n[API Error - Attempt {attempt + 1}] {e}")
            time.sleep(1)  # 出错歇一秒

    return None


def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    print(f"📖 读取 {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if MAX_SAMPLES:
        lines = lines[:MAX_SAMPLES]
        print(f"⚠️ 测试模式：只处理前 {MAX_SAMPLES} 条数据")
    else:
        print(f"🚀 全量模式：将处理所有 {len(lines)} 条数据")

    # 检查断点续传
    start_idx = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            start_idx = len(f.readlines())
            if start_idx > 0:
                print(f"🔄 断点续传：跳过前 {start_idx} 条")

    # 打开文件准备写入
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        # 使用 tqdm 显示进度条
        for i, line in tqdm(enumerate(lines), total=len(lines), initial=start_idx):
            if i < start_idx: continue

            item = json.loads(line)
            question = item['question']

            sub_questions = get_decomposition(client, question)

            # 验证逻辑
            is_valid = False
            if sub_questions and isinstance(sub_questions, list) and len(sub_questions) >= 2:
                # 只要列表中有字符串包含 PREV_ANSWER 即可，位置不限（有时模型会分3步）
                if any("PREV_ANSWER" in str(sq) for sq in sub_questions):
                    is_valid = True

            if is_valid:
                item['decomposition'] = sub_questions
                item['attackable'] = True
            else:
                item['decomposition'] = None
                item['attackable'] = False

            f_out.write(json.dumps(item) + "\n")
            f_out.flush()

            # 💤 关键修改：每条请求后暂停 0.1 秒，防止触发 API 速率限制 (QPS Limit)
            time.sleep(0.1)

    print(f"\n✅ 数据分解完成！输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()