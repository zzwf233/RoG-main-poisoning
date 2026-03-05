import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable


class PromptBuilder(object):
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""

    def __init__(self, prompt_path, add_rule=False, use_true=False, cot=False, explain=False, use_random=False,
                 each_line=False, maximun_token=4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line

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
        graph = utils.build_graph(question_dict['graph'])
        entities = question_dict['q_entity']
        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction

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
                graph = utils.build_graph(graph_data)
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
                context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            else:
                context = ""

            input = self.GRAPH_CONTEXT.format(context=context) + input

        input = self.prompt_template.format(instruction=instruction, input=input)

        return input

    def check_prompt_length(self, prompt, list_of_paths, maximun_token):
        '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths
        else:
            # Shuffle the paths
            random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)
            return "\n".join(new_list_of_paths)
