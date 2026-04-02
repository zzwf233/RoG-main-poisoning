import networkx as nx
from collections import deque


# import walker # 如果你没有安装 walker 库且不运行训练代码，可以将此行注释掉，否则保留

def build_graph(graph: list, bidirectional: bool = False) -> nx.DiGraph:
    G = nx.DiGraph()
    for triplet in graph:
        # --- 修改 1: 增加数据格式兼容性 (List vs Dict) ---
        if isinstance(triplet, dict):
            # 处理字典格式 {'head': '...', 'relation': '...', 'tail': '...'}
            h = triplet.get('head') or triplet.get('h')
            r = triplet.get('relation') or triplet.get('r')
            t = triplet.get('tail') or triplet.get('t')
        elif len(triplet) == 3:
            # 处理标准列表格式 [h, r, t]
            h, r, t = triplet
        else:
            continue  # 跳过格式错误的数据

        # 确保数据不是 None
        if h and r and t:
            rel = r.strip()
            G.add_edge(h, t, relation=rel)
            if bidirectional:
                G.add_edge(t, h, relation=rel)
    return G

# 定义一个函数来进行宽度优先搜索
def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    result_paths = []
    # 使用队列存储待探索节点和对应路径
    queue = deque([(start_node, [])])

    # 增加一个 visited 集合防止环路导致的死循环 (虽然 strict rule matching 通常不会死循环，但加一层保险)
    # 格式: (node, depth)
    visited = set([(start_node, 0)])

    while queue:
        current_node, current_path = queue.popleft()
        current_depth = len(current_path)

        # 如果当前路径符合规则，将其添加到结果列表中
        if current_depth == len(target_rule):
            result_paths.append(current_path)

            # --- 修改 2: 必须取消注释这个 Break！---
            # 对于推理攻击，我们只需要找到几条路径即可。
            # 如果不限制，对于某些连通度高的节点，程序会卡死。
            if len(result_paths) >= max_p:
                break
            continue  # 找到一条路径后，这条路径就不需要继续深挖了

        # 如果当前路径长度小于规则长度，继续探索
        if current_depth < len(target_rule):
            if current_node not in graph:
                continue

            for neighbor in graph.neighbors(current_node):
                # 剪枝：如果当前边类型与规则中的对应位置不匹配，不继续探索该路径
                edge_data = graph[current_node][neighbor]

                # 兼容不同图实现/版本下的 edge_data 形态：
                # 1) {'relation': 'r'}
                # 2) 'r'（某些自定义/历史序列化图）
                # 3) Multi(Graph/DiGraph) 形态 {key: {'relation': 'r'}}
                rel = None
                if isinstance(edge_data, dict):
                    if 'relation' in edge_data:
                        rel = edge_data.get('relation')
                    else:
                        # MultiGraph: 取第一条边上的 relation
                        for _, attrs in edge_data.items():
                            if isinstance(attrs, dict) and 'relation' in attrs:
                                rel = attrs.get('relation')
                                break
                            if isinstance(attrs, str):
                                rel = attrs
                                break
                elif isinstance(edge_data, str):
                    rel = edge_data

                # 检查关系是否匹配
                target_rel = target_rule[current_depth]
                if rel != target_rel:
                    continue

                # 防止往回走 (简单的环路检测)
                # 如果你想允许环路 (A->B->A)，可以去掉下面这行，但在知识图谱推理中通常不需要回头路
                if neighbor in [x[0] for x in current_path]:
                    continue

                # 加入队列
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
