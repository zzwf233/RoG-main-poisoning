# RAG Safety 论文的实验配置要点与“达不到指标”的最可能原因排查

你问的核心其实是两件事：

- **RAG Safety 论文里 RoG（Clean, CWQ）到底是怎么跑出来的？配置是什么？**
- **你现在 Table3（CWQ / RoG / Clean）只有 44.27, 38.86, 41.22, 40.36, 41.06, 40.33（3531 样本），是不是“配置不对”导致比论文低一截？**

结论先说：**很大概率确实是配置/实现偏差导致**，而且从你仓库当前实现来看，存在几个“会把 RoG 从多跳推理退化成近似 1-hop/噪声检索”的点；这类偏差非常容易把 CWQ 的 Clean F1 从论文水平（≈55）拉到你现在的 ≈39。

---

## RAG Safety 论文明确写了哪些实验配置

下面内容来自你上传的 **`/mnt/data/RAG Safety.pdf`**（我按 PDF 里的章节与表格直接摘取要点，便于你核对）。

### 论文的 Table3：CWQ / RoG / Clean 的目标指标

在 **Table 3（KGQA performance under data poisoning attack）** 中，RoG 在 **CWQ / Clean** 的 6 个指标为：

- **Hit = 64.94**
- **F1 = 54.86**
- **Precision = 56.64**
- **Recall = 57.57**
- **Hits@1 = 56.87**
- **EM = 47.18**

这就是你想对齐的“50%~60%区间”的来源（至少对 F1/Hits@1/Precision/Recall 来说确实在 50 左右，Hit 更高）。

> 你当前结果（Hit 44.27 / F1 38.86 / …）与论文的 CWQ Clean 相比，差距大约是：  
> **Hit -20.7、F1 -16.0、Hits@1 -15.8、EM -6.9**（按同顺序粗算）。  
> 这不像“随机波动”，更像系统性退化（模型/检索/规则/提示/评估口径等链路问题）。

### 论文 Section 4.3：实现细节里最关键的三件事

在 **4.3 Baseline and Implementation Details** 里，论文给了很关键的设置：

1) **对抗答案生成用 GPT-4**，并且多轮生成，得到 **N = 5** 个候选对抗答案。  
2) **关系路径（relation paths）生成用 LLMRoG**：  
   - 这是一个公开的 **KG-specific language model**，**基于 LLaMA-2-7B-hf**，并且专门用于生成能在 KG 中 ground 的 relation paths。  
3) 攻击预算：每个对抗答案 **K = 4** 条扰动 triple（因此每题上限 **N×K = 20** 条插入 triple）。

> 这段虽然写的是 attack setting，但它间接说明：**他们的 planning（relation path generation）阶段是靠“KG-specific 的 RoG 模型”完成的**，不是随便一个 Llama，也不是“把 relation token 随便 regex 抓出来就当 path”。

---

## 你达不到指标，最可能是哪些“配置不对 / 实现偏差”

我把“最可能把 CWQ Clean 拉低 10–20 个点”的问题，按优先级列成一个非常具体的排查表（每条都能快速验证）。

### 优先级最高：你的 planning 输出在仓库里很可能被“退化成 1-hop 规则集合”

在你仓库当前实现里（RoGplanning：`src/qa_prediction/gen_rule_path.py`；推理：`src/qa_prediction/build_qa_input.py` + `src/utils/graph_utils.py`），存在一个结构性风险：

- `gen_rule_path.py` 的 `parse_prediction()` **不是解析“relation path 序列”**，而是用 regex 把输出里所有像 `a.b.c` 的片段都抓出来，然后 `set()` 去重后丢到 `rules` 里。  
  这会把原本应该是 `r1 -> r2 -> r3` 的**多跳路径**，直接变成**一堆彼此独立的单条 relation 名**。
- `PromptBuilder.apply_rules()` 里有一段“兼容字符串规则：将单关系字符串视为 1-hop 规则”，会把每个字符串 relation 当成长度为 1 的 rule 来 BFS。  
  于是你系统在 clean 评测时，很可能变成：  
  **“对每个问题实体，用若干个 1-hop relation 各跑一遍 BFS，然后把得到的一堆 1-hop reasoning paths 拼到 prompt 里”**  
  ——这与论文的 **planning-retrieval-reasoning** 多跳推理范式差很大（CWQ 恰好 2-hop/3-hop 占比高，因此会掉分尤其明显）。

✅ 快速验证方式（10 分钟内能做）  
从你的 cwq 规则文件里随机抽 20 条样本，看 `rules` 字段是否类似下面两类：

- **正确形态（应该有）**：`[["r1","r2"], ["r3","r4","r5"], ...]` 或者至少能还原成序列  
- **退化形态（你现在很可能是）**：`["people.person.nationality","location.country.languages_spoken", ...]`（只有单 relation，没有路径结构）

如果是第二种，那基本就解释了你为何比论文低一大截：CWQ 多跳问题直接被削弱了。

### 第二优先：你当前 graph 是无向图，会引入大量“方向错误的 reasoning path”

你仓库 `build_graph()` 用的是 `nx.Graph()`（无向）；而 Freebase 三元组（head, relation, tail）语义上是有方向的。无向化会带来两类后果：

- **错误路径激增**：你可以沿着“反方向”走，但 relation 名还是原来那条，这在语义上不成立；
- **prompt 噪声上升**：更多路径 → 更长 prompt → 更容易触发“关键信息被淹没 / lost in the middle”类效应（RAG 场景里很常见，论文也强调检索阶段质量至关重要）。

这类问题常见现象是：Hit/Hits@1/F1 都会被拖累（尤其 Hits@1）。

✅ 快速验证方式  
对同一个样本，统计：
- 规则 grounding 后的 reasoning paths 数量分布（平均每条 rule 产生多少 path）
- prompt 截断率（`check_prompt_length()` 实际保留了多少行）
如果 reasoning paths 非常多、保留很少（大量截断），说明你在“错误膨胀”里。

### 第三优先：你 `build_qa_input.py` 在 master 分支是**会语法报错**的状态

你当前 master 的 `src/qa_prediction/build_qa_input.py` 存在明显缩进错误（`if len(rules) > 0:` 和 `check_prompt_length` 里 `else:` 下的块），按 Python 语法会直接报错、整个推理跑不起来。

你既然已经跑出了 3531 的结果，很可能是你本地已修过、但仓库未更新；**如果你有多人协作或分机器跑，可能不同环境用的是不同版本文件**，这会导致你以为“配置不对”，其实是“代码版本漂移”。

✅ 快速验证方式  
在你跑 Table3 的同一环境里执行：

- `python -m py_compile src/qa_prediction/build_qa_input.py`

如果报错，那么你现在用于跑分的并不是这个版本（或路径不同）。

### 第四优先：生成端（Llama pipeline）的解码参数默认值不透明，可能导致结果不稳定/偏差

你 `src/llms/language_models/llama.py` 用 HF `pipeline("text-generation")` 推理，但调用 `self.generator(...)` 时没有显式指定 `do_sample/temperature/top_p/num_beams` 等关键解码参数。

不同 transformers 版本/不同 pipeline 默认值可能不同，这会造成：

- 输出格式更啰嗦（影响 Precision/EM）
- 结果随机性更强（影响 Hits@1/F1）

这通常不会单独造成 -15 点，但会雪上加霜。

✅ 快速验证方式  
固定随机种子（torch + transformers）后连跑 2 次 CWQ clean 的 200 条子集，如果指标波动 >1–2 个点，说明解码随机性没被控制住。

---

## “是不是配置不对”——我建议你优先对齐的论文配置清单

为了最大概率把 CWQ Clean 拉回到论文区间（F1≈55、Hits@1≈57），我建议你按下面顺序做“对齐实验”（不用先大改模型）：

1) **确认你使用的确实是 RoG 的 KG-specific 模型权重（LLMRoG 体系）**  
   - 论文 relation path generation 用的是 LLMRoG（KG-specific）。  
   - 你仓库里 `model_name=RoG` 实际映射到 `Llama` wrapper（属于正常设计），但关键是：`--model_path` 指向的权重必须是真正的 RoG（KG-specific/finetuned）而不是 base Llama。  
2) **把 planning 输出从“relation 集合”恢复成“relation path 序列”**（这是最大增益点）  
   - 让 rules 是 list-of-list（1-hop/2-hop/3-hop）。  
3) **图结构改为有向（或显式加 inverse relation）**，降低噪声  
4) **固定解码为 deterministic（do_sample=False）**，先追求稳定与格式收敛  
5) 再做 prompt/路径排序/候选答案选择等增强

---

## 如果你愿意给我两项信息，我可以更快判断是否“配置不对”

你只要补充下面两点（非常关键）：

1) 你跑 Table3（CWQ Clean）时，`MODEL_PATH / PRED_MODEL_PATH / RULE_MODEL_PATH` 实际分别指向什么（本地目录还是 HF 缓存）？你用的是 RoG 权重还是 Llama 基座？  
2) 你生成的 rule 文件里，`rules` 字段是“单 relation 字符串列表”还是“多跳 relation path（list-of-list）”？

有了这两点，我基本可以直接判断：你现在的 40% 是“配置不对”还是“实现退化”，以及哪一条改动最可能一次性带来 +10~15 点提升。
