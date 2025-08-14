import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, List, Optional, TypedDict

# ---- 类型别名（不影响逻辑，仅让代码更清晰）----
Edge = Tuple[int, int]
CostMap = Dict[Edge, float]
CutMap = Dict[Edge, int]

class Best(TypedDict, total=False):
    obj: float
    cut: CutMap
    count: int
    path: List[str]     # 新增：记录生成 clusters 的 JOIN/CUT 路径


def contract_and_merge_costs(graph: nx.Graph, costs: CostMap, a: int, b: int, cut_edges: CutMap, log: bool = False):
    assert graph.has_node(a) and graph.has_node(b), \
        f"Edge ({a},{b}) not consistent with current graph."

    # Step 1: prepare new cut_edges
    new_cut_edges = cut_edges.copy()

    neighbors = set(graph.neighbors(a)).union(graph.neighbors(b))
    neighbors.discard(a)
    neighbors.discard(b)

    for c in neighbors:
        key_ac = (min(a, c), max(a, c))
        key_bc = (min(b, c), max(b, c))
        cut_bc = cut_edges.get(key_bc, -1)
        if cut_bc == 1:
            new_cut_edges[key_ac] = 1

    # Step 2: prepare new_costs BEFORE merge
    new_costs: CostMap = {}
    touched = set()

    for c in neighbors:
        key_ac = (min(a, c), max(a, c))
        key_bc = (min(b, c), max(b, c))
        cost_a = costs.get(key_ac, 0.0)
        cost_b = costs.get(key_bc, 0.0)
        new_costs[key_ac] = cost_a + cost_b
        touched.add(key_ac)

    # keep other costs unchanged
    for (u, v), w in costs.items():
        key = (min(u, v), max(u, v))
        if key not in touched and b not in key:
            new_costs[key] = w

    # Step 3: merge
    new_graph = nx.contracted_nodes(graph, a, b, self_loops=False)

    return new_graph, new_costs, new_cut_edges


def propagate_zero_labels(cut_edges: CutMap, u: int, v: int, costs: CostMap, log: bool = False) -> CutMap:
    if log:
        print(f"[PROP_ZERO] Start propagate_zero_labels from edge ({u}, {v})")
        print("  - Current cut_edges, costs:")
        for edge in sorted(cut_edges):
            label = cut_edges[edge]
            cost_str = f",  cost: {costs[edge]:.2f}" if edge in costs else ""
            print(f"    {edge}: {label}{cost_str}")

    # Step 1: Build adjacency map of all edges labeled as uncut (0)
    uncut_adj: Dict[int, set] = {}
    for (a, b), val in cut_edges.items():
        if val == 0:
            uncut_adj.setdefault(a, set()).add(b)
            uncut_adj.setdefault(b, set()).add(a)

    # Step 2: BFS to find all nodes reachable via uncut edges
    visited: set = set()
    queue = deque([u, v])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in uncut_adj.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)

    # Step 3: For each pair of visited nodes, propagate 0-label if edge was undecided
    visited_list = list(visited)
    for i in range(len(visited_list)):
        for j in range(i + 1, len(visited_list)):
            n1, n2 = visited_list[i], visited_list[j]
            edge = (min(n1, n2), max(n1, n2))
            if edge in cut_edges and cut_edges[edge] == -1:
                cut_edges[edge] = 0
                if log:
                    found = "FOUND" if edge in costs else "NOT FOUND"
                    value = costs.get(edge, 0.0)
                    print(f"\033[96m  Propagate 0-label: edge {edge} with cost {value:.2f} ({found})\033[0m")
    return cut_edges


def is_feasible_cut(graph: nx.Graph, cut_edges: CutMap, verbose: bool = False) -> bool:
    # 1. 仅处理原始图中的边
    edges_to_cut = []
    for u, v in graph.edges:
        if cut_edges.get((u, v), 0) == 1 or cut_edges.get((v, u), 0) == 1:
            edges_to_cut.append((u, v))

    # 2. 拷贝图，删掉这些边
    g_copy = graph.copy()
    g_copy.remove_edges_from(edges_to_cut)

    # 3. 构造每个节点的连通分量编号
    components = list(nx.connected_components(g_copy))
    label: Dict[int, int] = {}
    for idx, comp in enumerate(components):
        for node in comp:
            label[node] = idx

    # 4. 验证所有 cut=1 的边是否真的跨分量
    for u, v in edges_to_cut:
        if label[u] == label[v]:
            if verbose:
                print(f"Edge ({u}, {v}) is cut but endpoints are still in same component.")
            return False

    # 5. 所有 cut 边都成功断开
    return True


def print_edge_labels_inline(graph: nx.Graph, cut_edges: CutMap, obj: float, best_obj: float) -> None:
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    print(f"[GRAPH STATUS] obj = {obj:.2f}, best = {best_obj:.2f}")
    parts = []
    for u, v in graph.edges():
        e = (min(u, v), max(u, v))
        label = cut_edges.get(e, -1)
        if label == 1:
            parts.append(f"{RED}{e}{RESET}")
        elif label == 0:
            parts.append(f"{GREEN}{e}{RESET}")
        else:
            parts.append(f"{e}")  # undecided = default color
    print("  Edges: " + "  ".join(parts))


def update_best_if_feasible_final(
    orig_graph: nx.Graph,
    cut_edges: CutMap,
    obj: float,
    best: Best,
    log: bool = False,
    path: Optional[List[str]] = None
) -> None:
    """
    新增 path: 打印并保存从根到当前解的 JOIN/CUT 决策序列。
    """
    if is_feasible_cut(orig_graph, cut_edges):
        # 构造 clusters 字符串
        g_copy = orig_graph.copy()
        edges_to_cut = [(u, v) for (u, v), val in cut_edges.items() if val == 1]
        g_copy.remove_edges_from(edges_to_cut)
        clusters = list(nx.connected_components(g_copy))
        clusters_str = ' '.join(
            '{' + ','.join(str(node) for node in sorted(comp)) + '}' for comp in clusters
        )
        path_str = " -> ".join(path or ["Start"])

        if obj > best.get('obj', 0.0):
            best['obj'] = obj
            best['cut'] = cut_edges
            best['count'] = 1
            best['path'] = list(path or ["Start"])
            if log:
                print(f"[UPDATE] New best obj = {obj:.2f}")
                print(f"        Clusters: {clusters_str}")
                print(f"        Path:     {path_str}")
        elif obj == best.get('obj', 0.0):
            # 并列最优也更新 path，便于追踪
            best['cut'] = cut_edges
            best['count'] = best.get('count', 0) + 1
            best['path'] = list(path or ["Start"])
            if log:
                print(
                    f"[TIE] Another feasible cut with obj = {obj:.2f}, "
                    f"Clusters: {clusters_str}, "
                    f"Path: {path_str}, total count = {best['count']}"
                )


bound_trace: List[Tuple[int, float, float]] = []  # (depth, tighter_bound, naive_bound)


def compute_tight_upper_bound(graph: nx.Graph, costs: CostMap, cut_edges: CutMap, max_cycle_length: int = 6) -> float:
    graph_edges = set(graph.edges())
    E_plus = {e for e, w in costs.items() if w > 0 and e in graph_edges and cut_edges.get(e, -1) != 1}
    E_minus = {e for e, w in costs.items() if w < 0 and e in graph_edges and cut_edges.get(e, -1) != 1}
    G_plus = graph.edge_subgraph(E_plus).copy()

    conflicted_cycles: List[List[Edge]] = []
    for (u, v) in E_minus:
        if u not in G_plus or v not in G_plus:
            continue
        try:
            path = nx.shortest_path(G_plus, u, v)
            if len(path) + 1 <= max_cycle_length:
                cycle_edges = [(min(path[i], path[i + 1]), max(path[i], path[i + 1])) for i in range(len(path) - 1)]
                rep_edge = (min(u, v), max(u, v))
                cycle_edges.append(rep_edge)
                if rep_edge in graph_edges and cut_edges.get(rep_edge, -1) != 1:
                    conflicted_cycles.append(cycle_edges)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    used_edges: set = set()
    reduce_total = 0.0
    for cycle in sorted(conflicted_cycles, key=lambda cyc: len(cyc)):
        if any(e in used_edges or cut_edges.get(e, -1) == 1 or e not in graph_edges for e in cycle):
            continue
        pos_edges = [e for e in cycle if e in E_plus and cut_edges.get(e, -1) != 1 and e in graph_edges]
        if not pos_edges:
            continue
        delta = min(costs[e] for e in pos_edges)
        reduce_total += delta
        used_edges.update(cycle)

    naive_upper_bound = sum(w for e, w in costs.items() if w > 0 and e in graph_edges and cut_edges.get(e, -1) != 1)
    return naive_upper_bound - reduce_total


def plot_bound_trace():
    if not bound_trace:
        print("[WARN] No bound trace to plot.")
        return
    depths, t_bounds, n_bounds = zip(*bound_trace)
    plt.figure(figsize=(10, 4))
    plt.plot(depths, t_bounds, label="Tight Bound", marker='o')
    plt.plot(depths, n_bounds, label="Naive Bound", linestyle='--', marker='x')
    plt.xlabel("Recursion Depth")
    plt.ylabel("Upper Bound Estimate")
    plt.title("Bound Comparison Per Branch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 🔧 Clear after plotting
    bound_trace.clear()


def print_edge_label_groups(cut_edges: CutMap, tag: str = ""):
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    cut = {e for e, v in cut_edges.items() if v == 1}
    uncut = {e for e, v in cut_edges.items() if v == 0}
    undecided = {e for e, v in cut_edges.items() if v == -1}

    print(f"[EDGE LABELS{f' @ {tag}' if tag else ''}]")
    print(f"  cut edges:      {RED}{cut}{RESET}")
    print(f"  uncut edges:    {GREEN}{uncut}{RESET}")
    print(f"  undecided edges:{undecided}")


def bnb_multicut(
        graph: nx.Graph,
        costs: CostMap,
        cut_edges: CutMap,
        obj: float,
        best: Best,
        log: bool,
        use_tight_bound: bool = True,
        depth: int = 0,
        node_counter: Optional[Dict[str, int]] = None,
        orig_graph: Optional[nx.Graph] = None,
        path: Optional[List[str]] = None):

    if node_counter is not None:
        node_counter['count'] += 1  # 🔢 每次进入一个分支节点就 +1

    if path is None:
        path = ["Start"]

    if not costs:
        if log:
            print(f"[NO COSTS] \033[92mcluster_obj={obj:.2f}\033[0m, best_obj={best.get('obj', 0.0):.2f}")
        cut_edges_copy = cut_edges.copy()
        for e in cut_edges_copy:
            if cut_edges_copy[e] == -1:
                cut_edges_copy[e] = 1
        # ➕ 传入 path
        update_best_if_feasible_final(orig_graph, cut_edges_copy, obj, best, True, path)
        return None

    # Compute the optimistic bound
    if use_tight_bound:
        bound = compute_tight_upper_bound(graph, costs, cut_edges)
    else:
        graph_edges = set(graph.edges())
        bound = sum(w for e, w in costs.items() if w > 0 and e in graph_edges and cut_edges.get(e, -1) != 1)

    if log:
        print(
            f"[ENTER BnB] \033[92mcluster_obj={obj:.2f}\033[0m, \033[91mbound={bound:.2f}\033[0m, best_obj={best.get('obj', 0.0):.2f}")
        naive = sum(w for e, w in costs.items() if w > 0 and e in graph.edges and cut_edges.get(e, -1) != 1)
        bound_trace.append((depth, bound, naive))
        print(f"\033[93m[BOUND] tighter = {bound:.2f}, naive = {naive:.2f}, Δ = {naive - bound:.2f}\033[0m")

    if obj + bound < best.get('obj', 0.0):
        if log:
            print(f"[PRUNE] Max possible obj = {obj + bound:.2f} < best obj = {best.get('obj', 0.0):.2f} → prune branch")
        return None

    edge, max_cost = max(costs.items(), key=lambda item: item[1])
    u, v = edge
    edge_key = (min(u, v), max(u, v))

    # Join branch
    if log:
        print_edge_label_groups(cut_edges, f"before JOIN ({u},{v})")
    skip_join = max_cost < 0
    if skip_join:
        graph_join = None
    else:
        graph_join, costs_join, cut_edges_join = contract_and_merge_costs(graph, costs, u, v, cut_edges, log=log)
    if graph_join is not None:
        cut_edges_join[edge_key] = 0
        cut_edges_join = propagate_zero_labels(cut_edges_join, u, v, costs, log)
        obj_join = obj + max_cost
        if log:
            print(f"[BRANCH] Join: merging ({u}, {v}) with cost {max_cost:.2f}")
            print(f"  - New objective: {obj_join:.2f}")
        # ➕ 路径增加 JOIN
        bnb_multicut(graph_join, costs_join, cut_edges_join, obj_join, best, log, use_tight_bound, depth + 1,
                     node_counter, orig_graph, path + [f"JOIN({u},{v})"])
        if log:
            print_edge_label_groups(cut_edges_join, f"after JOIN ({u},{v})")

    # Cut branch
    if log:
        print_edge_label_groups(cut_edges, f"before CUT ({u},{v})")
    graph_cut = graph.copy()
    graph_cut.remove_edge(u, v)
    costs_cut = {e: w for e, w in costs.items() if e != edge}
    cut_edges_cut = cut_edges.copy()
    cut_edges_cut[edge_key] = 1
    if log:
        print(f"[BRANCH] Cut: removing edge {edge_key} with cost {max_cost:.2f}, Objective unchanged: {obj:.2f}")
    # ➕ 路径增加 CUT
    bnb_multicut(graph_cut, costs_cut, cut_edges_cut, obj, best, log, use_tight_bound, depth + 1,
                 node_counter, orig_graph, path + [f"CUT({u},{v})"])
    if log:
        print_edge_label_groups(cut_edges_cut, f"after CUT ({u},{v})")


def benchmark_solver(graph: nx.Graph, costs: CostMap, log: bool = False):
    print("[BENCHMARK] Running with naive bound...")
    solver_naive = BnBSolver(graph, costs, log=log, use_tight_bound=False)
    start_naive = time.time()
    _, obj_naive, count_naive = solver_naive.solve()
    time_naive = time.time() - start_naive

    print("[BENCHMARK] Running with tight bound...")
    solver_tight = BnBSolver(graph, costs, log=log, use_tight_bound=True)
    start_tight = time.time()
    _, obj_tight, count_tight = solver_tight.solve()
    time_tight = time.time() - start_tight

    print("\n[RESULT SUMMARY]")
    print(f"Naive Bound:  obj = {obj_naive:.2f}, time = {time_naive:.2f}s, nodes = {count_naive}")
    print(f"Tight Bound:  obj = {obj_tight:.2f}, time = {time_tight:.2f}s, nodes = {count_tight}")
    print(f"\033[93mSpeedup = {time_naive / time_tight:.2f}x, Δ obj = {obj_tight - obj_naive:.2f}\033[0m")

    plot_bound_trace()


class BnBSolver:
    def __init__(self, graph: nx.Graph, costs: CostMap, log: bool = True, use_tight_bound: bool = True):
        self.graph = graph
        self.costs = costs
        self.log = log
        self.use_tight_bound = use_tight_bound

    def solve(self):
        if self.log:
            print(f"graph for bnb solver:")
        normalized_costs: CostMap = {}
        for (u, v), w in self.costs.items():
            key = (min(u, v), max(u, v))
            normalized_costs[key] = w
            if self.log:
                print(f"{key}: {w:.2f}")
        cut_edges: CutMap = {e: -1 for e in normalized_costs}
        best: Best = {'obj': 0.0, 'cut': cut_edges, 'count': 0, 'path': ["Start"]}

        node_counter = {'count': 0}

        start = time.time()
        bnb_multicut(
            self.graph,
            normalized_costs,
            cut_edges,
            obj=0.0,
            best=best,
            log=self.log,
            use_tight_bound=self.use_tight_bound,
            node_counter=node_counter,
            orig_graph=self.graph,
            path=["Start"]
        )
        end = time.time()

        if self.log:
            print("Cut edges:")
            print(f"[FINISH] total time = {end - start:.2f} seconds")
            print(f"[STATS] Total nodes visited in BnB: {node_counter['count']}")
            print(f"[BEST PATH] {' -> '.join(best.get('path', []))}")
        obj = 0.0
        for u, v in self.graph.edges():
            e = (min(u, v), max(u, v))
            if best['cut'].get(e, -1) == 1:
                cost = normalized_costs.get(e, 0.0)
                obj += cost
        if self.log:
            print("[DEBUG] Final raw best cut:", best['cut'])
        return best['cut'], obj, best['count']