import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
import re
from typing import Callable


class PromptBuilder(object):
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, answer the question with entities only. Return only the final answers (one per line), with no explanation."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""

    def __init__(self, prompt_path, add_rule=False, use_true=False, cot=False, explain=False, use_random=False,
                 each_line=False, maximun_token=4096, tokenize: Callable = lambda x: len(x),
                 path_top_k: int = 20, bidirectional_graph: bool = False):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line
        self.path_top_k = max(0, int(path_top_k))
        self.bidirectional_graph = bidirectional_graph

    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template

    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                # 兼容字符串规则：将单关系字符串视为 1-hop 规则
                if isinstance(rule, str):
                    rule = [rule.strip()]
                if not isinstance(rule, list) or len(rule) == 0:
                    continue
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results

    def direct_answer(self, question_dict):
        graph = utils.build_graph(question_dict['graph'], bidirectional=self.bidirectional_graph)
        entities = question_dict['q_entity']
        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
    
    def prioritize_paths_for_poison(self, lists_of_paths, poison_target: str, poisoned_only: bool = False):
        """Prioritize (or isolate) paths that explicitly mention poison target."""
        target = str(poison_target).strip().lower()
        if not target:
            return lists_of_paths

        hit_paths = []
        miss_paths = []
        for p in lists_of_paths:
            ps = str(p).lower()
            if target in ps:
                hit_paths.append(p)
            else:
                miss_paths.append(p)

        if not hit_paths:
            return lists_of_paths

        if poisoned_only:
            return hit_paths
        return hit_paths + miss_paths
    def build_poison_hint_path(self, poison_target: str) -> str:
        target = str(poison_target).strip()
        if not target:
            return ""
        return f"Injected_Knowledge -> leads_to -> {target}"
    
    def extract_poison_evidence_lines(self, graph_data, poison_target: str, limit: int = 6):
        target = str(poison_target).strip().lower()
        if not target or not isinstance(graph_data, list):
            return []

        direct = []
        related_nodes = set()
        for triplet in graph_data:
            if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                continue
            h, r, t = triplet
            hs, rs, ts = str(h), str(r), str(t)
            hsl, tsl = hs.lower(), ts.lower()
            if target in hsl or target in tsl:
                direct.append(f"{hs} -> {rs} -> {ts}")
                related_nodes.add(hs)
                related_nodes.add(ts)

        if not direct:
            return []

        # Add 1-hop structural neighbors around direct poison lines for better plausibility.
        expanded = list(direct)
        for triplet in graph_data:
            if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                continue
            h, r, t = map(str, triplet)
            if h in related_nodes or t in related_nodes:
                line = f"{h} -> {r} -> {t}"
                if line not in expanded:
                    expanded.append(line)
            if len(expanded) >= limit:
                break

        return expanded[:limit]
    def process_input(self, question_dict):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']

        if not question.endswith('?'):
            question += '?'

        if self.add_rule:
            # 🔍 修改 1: 安全获取 graph，防止 key error
            graph_data = question_dict.get('graph', [])
            if graph_data:
                graph = utils.build_graph(graph_data, bidirectional=self.bidirectional_graph)
            else:
               graph = {}

            # 🔍 修改 2: 安全获取 q_entity
            entities = question_dict.get('q_entity', [])
            if isinstance(entities, str):
                entities = [entities]

            # 🔍 修改 3: 移除了之前重复的 if/else 代码块，保留一份逻辑
            if self.use_true:
                rules = question_dict.get('ground_paths', [])
            elif self.use_random:
                # 如果是 random，需要现场在图上跑
                if graph and entities:
                    _, rules = utils.get_random_paths(entities, graph)
                else:
                   rules = []
            else:
                rules = question_dict.get('predicted_paths', [])

            lists_of_paths = []
            if len(rules) > 0:
            # 🔍 修改 4 (核心): 投毒适配逻辑
            # 检查 rules 是否已经是生成的路径字符串（即包含 "->"），如果是，说明已经 Grounding 过了
            # 这是为了兼容 predict_with_grounding.py 的输出
                is_already_grounded = False
                if isinstance(rules[0], str) and " -> " in rules[0]:
                    is_already_grounded = True

                if is_already_grounded:
                # 如果已经是路径字符串，直接使用，不要再跑 BFS
                    lists_of_paths = rules
                else:
                # 否则，视为规则，执行 BFS 检索
                    if graph:  # 只有有图才能检索
                        reasoning_paths = self.apply_rules(graph, rules, entities)
                        lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]

            # context = "\n".join(lists_of_paths) # 原代码注释掉的

            poison_target = question_dict.get("poison_target", question_dict.get("poison_target_entity", ""))
            is_poisoned = bool(question_dict.get("is_poisoned", False))

            if lists_of_paths and poison_target:
                # Step1: 先按 target 命中重排（命中在前）
                lists_of_paths = self.prioritize_paths_for_poison(
                    lists_of_paths,
                    poison_target=poison_target,
                    poisoned_only=False,   # 先不要直接只保留，避免过度激进
                )

            # Step2: poisoned 样本做“硬优先 top-k”
            # 目的：在不改 rule 的前提下，让同则命中的 target-path 不被大量 clean-path 淹没
            if is_poisoned and poison_target:
                t = str(poison_target).strip().lower()
                hit_paths = [p for p in lists_of_paths if t and t in str(p).lower()]
                miss_paths = [p for p in lists_of_paths if not (t and t in str(p).lower())]

                # 你可以调这个比例：先给 4 条命中 + 2 条非命中
                k_hit = 4
                k_miss = 2

                if hit_paths:
                    lists_of_paths = hit_paths[:k_hit] + miss_paths[:k_miss]

                # Step3: 若仍没有命中路径，再回退到 graph 证据 / hint
                if not hit_paths:
                    evidence_lines = self.extract_poison_evidence_lines(question_dict.get("graph", []), poison_target)
                    if evidence_lines:
                        # graph证据放最前，增强可见性
                        lists_of_paths = evidence_lines[:k_hit] + miss_paths[:k_miss]
                    else:
                        hint = self.build_poison_hint_path(poison_target)
                        if hint:
                            lists_of_paths = [hint] + miss_paths[:k_miss]
        input = self.QUESTION.format(question=question)

        # 🔍 修改 5: 安全获取 choices
        choices_list = question_dict.get('choices', [])

        # MCQ
        if len(choices_list) > 0:
            choices = '\n'.join(choices_list)
            input += self.CHOICES.format(choices=choices)
            if self.add_rule:
                instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ
        else:
            if self.add_rule:
                instruction = self.SAQ_RULE_INSTRUCTION
            else:
                instruction = self.SAQ_INSTRUCTION

        if self.cot:
             instruction += self.COT

        if self.explain:
            instruction += self.EXPLAIN

        if self.each_line:
            instruction += self.EACH_LINE

        if self.add_rule:
            other_prompt = self.prompt_template.format(instruction=instruction,
                                                   input=self.GRAPH_CONTEXT.format(context="") + input)

            # 只有当 lists_of_paths 不为空时才处理 context
            if lists_of_paths:
                priority_terms = [str(question_dict.get("poison_target", "")), str(question_dict.get("poison_target_entity", ""))]
                context = self.check_prompt_length(
                    other_prompt,
                    lists_of_paths,
                    self.maximun_token,
                    priority_terms=priority_terms,
                    question_text=question
                )
            else:
                context = ""

            input = self.GRAPH_CONTEXT.format(context=context) + input

        input = self.prompt_template.format(instruction=instruction, input=input)

        return input

    def check_prompt_length(self, prompt, list_of_paths, maximun_token, priority_terms=None, question_text=""):
        """先排序后截断：提高证据密度"""
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths

        priority_terms = [str(x).strip().lower() for x in (priority_terms or []) if str(x).strip()]

        # 非停用词关键词（很轻量）
        stop = {"the","a","an","is","are","was","were","in","on","at","of","to","for","and","or","what","which","who","where","when","how","does","did","do"}
        q_terms = [w for w in re.findall(r"[a-z0-9_]+", str(question_text).lower()) if w not in stop and len(w) > 2]
        q_terms = set(q_terms)

        # 高频关系惩罚（可以后续统计替换）
        high_freq_rel_hints = {"type.object.type", "common.topic.notable_types"}

        seen = set()
        unique_paths = []
        for p in list_of_paths:
            ps = str(p).strip()
            if ps not in seen:
                seen.add(ps)
                unique_paths.append(ps)

        def path_score(path_str: str) -> float:
            s = path_str.lower()
            score = 0.0

            # 1) poison/priority 命中
            for t in priority_terms:
                if t in s:
                    score += 8.0

            # 2) question 关键词命中
            qt_hit = sum(1 for t in q_terms if t in s)
            score += qt_hit * 1.5

            # 3) 关系数量（太短偏泛化）
            rels = [x.strip() for x in s.split("->")]
            score += min(len(rels), 4) * 0.8
            if len(rels) <= 1:
                score -= 1.5

            # 4) 高频泛化关系惩罚
            for h in high_freq_rel_hints:
                if h in s:
                    score -= 1.0

            # 5) 超长路径轻惩罚
            score -= max(0, len(path_str) // 180) * 0.2

            return score

        ranked_paths = sorted(unique_paths, key=path_score, reverse=True)
        if self.path_top_k > 0:
            ranked_paths = ranked_paths[:self.path_top_k]

        new_list_of_paths = []
        for p in ranked_paths:
            tmp_all_paths = "\n".join(new_list_of_paths + [p])
            tmp_all_tokens = prompt + tmp_all_paths
            if self.tokenize(tmp_all_tokens) > maximun_token:
                return "\n".join(new_list_of_paths)
            new_list_of_paths.append(p)
        return "\n".join(new_list_of_paths)