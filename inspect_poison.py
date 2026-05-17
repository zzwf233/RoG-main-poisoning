import json

# 文件路径
input_file = "datasets/poisoned_cwq_full.jsonl"


def print_example(item):
    print("=" * 60)
    print(f"🎯 问题 (Question): {item['question']}")
    print(f"💡 原始真值 (Ground Truth): {item.get('answer', 'Unknown')}")
    print(f"☠️ 注入毒药 (Poison Target): {item.get('poison_target', 'None')}")

    print("-" * 20)
    print("🔍 毒药是如何注入的 (Injected Triples):")

    # 打印前 5 个被插入到图谱最前面的三元组（通常毒药都在最前面）
    graph = item.get('graph', [])
    count = 0
    for t in graph:
        # 只显示包含毒药的三元组
        if item.get('poison_target') in str(t):
            print(f"   {t[0]} --[{t[1]}]--> {t[2]}")
            count += 1
            if count >= 5: break

    if count == 0:
        print("   (在图谱前几行未发现显式毒药，可能挂载在深处)")
    print("=" * 60)
    print("\n")


# 读取并打印前 10 个成功投毒的例子
with open(input_file, 'r', encoding='utf-8') as f:
    valid_count = 0
    for line in f:
        item = json.loads(line)
        # 只看那些成功注入了毒药的样本
        if item.get('poison_target') and "Unknown" not in item['poison_target']:
            print_example(item)
            valid_count += 1
            if valid_count >= 5:  # 只看前 5 个，想看更多改这里
                break