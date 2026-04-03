import networkx as nx
from collections import deque


# import walker # 如果你没有安装 walker 库且不运行训练代码，可以将此行注释掉，否则保留

def build_graph(graph: list, bidirectional: bool = False) -> nx.DiGraph:
    G = nx.DiGraph()
    for triplet in graph:
        if isinstance(triplet, dict):
            h = triplet.get('head') or triplet.get('h')
            r = triplet.get('relation') or triplet.get('r')
            t = triplet.get('tail') or triplet.get('t')
        elif len(triplet) == 3:
            h, r, t = triplet
        else:
            continue
        if h and r and t:
            rel = r.strip()
            G.add_edge(h, t, relation=rel)
            if bidirectional:
                G.add_edge(t, h, relation=rel)
    return G

# 定义一个函数来进行宽度优先搜索
def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    result_paths = []
    queue = deque([(start_node, [])])
    visited = set([(start_node, 0)])

    while queue:
        current_node, current_path = queue.popleft()
        current_depth = len(current_path)

        if current_depth == len(target_rule):
            result_paths.append(current_path)
            if len(result_paths) >= max_p:
                break
            continue

        if current_depth < len(target_rule):
            if current_node not in graph:
                continue

            for neighbor in graph.neighbors(current_node):
                # 获取所有平行边
                edge_data_dict = graph.get_edge_data(current_node, neighbor)
                if edge_data_dict is None:
                    continue
                for key, edge_data in edge_data_dict.items():
                    rel = edge_data.get('relation')
                    if rel != target_rule[current_depth]:
                        continue
                    if neighbor in [x[0] for x in current_path]:
                        continue
                    new_path = current_path + [(current_node, rel, neighbor)]
                    queue.append((neighbor, new_path))
    return result_paths


# --- 以下函数通常用于训练数据生成，推理阶段很少用到，保持原样即可 ---

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p) - 1):
            u = p[i]
            v = p[i + 1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths


def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    '''
    Get all simple paths connecting question and answer entities within given hop
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append([(e[0], graph[e[0]][e[1]]['relation'], e[1]) for e in p])
    return result_paths


# 注意：下面的函数依赖 walker 库。如果你只跑推理攻击，且报错说找不到 walker，可以把下面删掉或注释掉
try:
    import walker


    def get_negative_paths(q_entity: list, a_entity: list, graph: nx.Graph, n_neg: int, hop=2) -> list:
        # sample paths
        start_nodes = []
        end_nodes = []
        node_idx = list(graph.nodes())
        for h in q_entity:
            if h in graph:
                start_nodes.append(node_idx.index(h))
        for t in a_entity:
            if t in graph:
                end_nodes.append(node_idx.index(t))
        paths = walker.random_walks(graph, n_walks=n_neg, walk_len=hop, start_nodes=start_nodes, verbose=False)
        # Add relation to paths
        result_paths = []
        for p in paths:
            tmp = []
            # remove paths that end with answer entity
            if p[-1] in end_nodes:
                continue
            for i in range(len(p) - 1):
                u = node_idx[p[i]]
                v = node_idx[p[i + 1]]
                tmp.append((u, graph[u][v]['relation'], v))
            result_paths.append(tmp)
        return result_paths


    def get_random_paths(q_entity: list, graph: nx.Graph, n=3, hop=2) -> tuple[list, list]:
        # sample paths
        start_nodes = []
        node_idx = list(graph.nodes())
        for h in q_entity:
            if h in graph:
                start_nodes.append(node_idx.index(h))
        paths = walker.random_walks(graph, n_walks=n, walk_len=hop, start_nodes=start_nodes, verbose=False)
        # Add relation to paths
        result_paths = []
        rules = []
        for p in paths:
            tmp = []
            tmp_rule = []
            for i in range(len(p) - 1):
                u = node_idx[p[i]]
                v = node_idx[p[i + 1]]
                tmp.append((u, graph[u][v]['relation'], v))
                tmp_rule.append(graph[u][v]['relation'])
            result_paths.append(tmp)
            rules.append(tmp_rule)
        return result_paths, rules
except ImportError:
    pass
