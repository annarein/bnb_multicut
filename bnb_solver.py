import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, List, Optional, TypedDict

"""
Multicut Branch-and-Bound (maximization form)

Edge labels in `CutMap`:
  1 = cut (different clusters)
  0 = uncut (same cluster)
 -1 = undecided

Objective:
  maximize sum of positive edge weights kept (or equivalently,
  maximize total gain from JOIN decisions). Upper bounds are
  computed via dual-feasible iterative cycle packing (ICP)
  over conflicted cycles (1 negative + positive path).

References for terminology and feasibility:
  - Minimum cost multicut / correlation clustering: see standard definitions.
  - Conflicted cycles & dual packing intuition: aligns with cycle-cover relaxations.
"""

# ---------- Type aliases ----------
Edge = Tuple[int, int]
CostMap = Dict[Edge, float]
CutMap = Dict[Edge, int]

class Best(TypedDict, total=False):
    """Container for the incumbent."""
    obj: float
    cut: CutMap
    count: int
    path: List[str]  # JOIN/CUT sequence from root to incumbent


# ---------- Graph update / feasibility helpers ----------

def contract_and_merge_costs(
    graph: nx.Graph,
    costs: CostMap,
    a: int,
    b: int,
    cut_edges: CutMap,
    log: bool = False
):
    """
    Contract nodes a,b; merge incident edge costs onto (a,·);
    inherit cut labels: if (b,c) was cut, then (a,c) becomes cut.

    Returns: (new_graph, new_costs, new_cut_edges)
    """
    assert graph.has_node(a) and graph.has_node(b), \
        f"Edge ({a},{b}) not consistent with current graph."

    # Cut labels after merge
    new_cut_edges = cut_edges.copy()
    neighbors = set(graph.neighbors(a)).union(graph.neighbors(b)) - {a, b}
    for c in neighbors:
        key_ac = (min(a, c), max(a, c))
        key_bc = (min(b, c), max(b, c))
        if cut_edges.get(key_bc, -1) == 1:
            new_cut_edges[key_ac] = 1

    # Costs after merge (sum on parallel edges)
    new_costs: CostMap = {}
    touched = set()
    for c in neighbors:
        key_ac = (min(a, c), max(a, c))
        key_bc = (min(b, c), max(b, c))
        new_costs[key_ac] = costs.get(key_ac, 0.0) + costs.get(key_bc, 0.0)
        touched.add(key_ac)

    # Keep other costs unchanged (skip edges incident to b)
    for (u, v), w in costs.items():
        key = (min(u, v), max(u, v))
        if key not in touched and b not in key:
            new_costs[key] = w

    # Perform contraction
    new_graph = nx.contracted_nodes(graph, a, b, self_loops=False)
    return new_graph, new_costs, new_cut_edges


def propagate_zero_labels(
    cut_edges: CutMap,
    u: int,
    v: int,
    costs: CostMap,
    log: bool = False
) -> CutMap:
    """
    Propagate label 0 (uncut) inside the connected subgraph
    formed by edges already labeled 0. Any undecided pair
    connected by a 0-labeled path becomes 0.
    """
    if log:
        print(f"[PROP_ZERO] Start propagate_zero_labels from edge ({u}, {v})")

    # Build adjacency from label-0 edges
    uncut_adj: Dict[int, set] = {}
    for (a, b), val in cut_edges.items():
        if val == 0:
            uncut_adj.setdefault(a, set()).add(b)
            uncut_adj.setdefault(b, set()).add(a)

    # BFS from {u,v} on 0-labeled edges
    visited: set = set()
    queue = deque([u, v])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(n for n in uncut_adj.get(node, []) if n not in visited)

    # Close under transitivity: mark undecided pairs among visited as 0
    visited_list = list(visited)
    for i in range(len(visited_list)):
        for j in range(i + 1, len(visited_list)):
            n1, n2 = visited_list[i], visited_list[j]
            e = (min(n1, n2), max(n1, n2))
            if e in cut_edges and cut_edges[e] == -1:
                cut_edges[e] = 0
                if log:
                    value = costs.get(e, 0.0)
                    print(f"\033[96m  Propagate 0-label: edge {e} (cost {value:.2f})\033[0m")
    return cut_edges


def is_feasible_cut(
    graph: nx.Graph,
    cut_edges: CutMap,
    verbose: bool = False
) -> bool:
    """
    Check multicut feasibility: removing edges with label 1
    must separate their endpoints into different components.
    """
    edges_to_cut = []
    for u, v in graph.edges:
        if cut_edges.get((u, v), 0) == 1 or cut_edges.get((v, u), 0) == 1:
            edges_to_cut.append((u, v))

    g_copy = graph.copy()
    g_copy.remove_edges_from(edges_to_cut)

    # Component map
    label: Dict[int, int] = {}
    for idx, comp in enumerate(nx.connected_components(g_copy)):
        for node in comp:
            label[node] = idx

    # Any cut edge whose endpoints stay connected → infeasible
    for u, v in edges_to_cut:
        if label[u] == label[v]:
            if verbose:
                print(f"Edge ({u}, {v}) is cut but endpoints remain connected.")
            return False
    return True


def print_edge_labels_inline(
    graph: nx.Graph,
    cut_edges: CutMap,
    obj: float,
    best_obj: float
) -> None:
    """Color-coded inline dump of edge labels and current objective."""
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
            parts.append(f"{e}")
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
    If the proposed labeling is feasible, update the incumbent and
    record the resulting clusters and decision path.
    """
    if is_feasible_cut(orig_graph, cut_edges):
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
            best['cut'] = cut_edges
            best['count'] = best.get('count', 0) + 1
            best['path'] = list(path or ["Start"])
            if log:
                print(
                    f"[TIE] Another feasible cut with obj = {obj:.2f}, "
                    f"Clusters: {clusters_str}, "
                    f"Path: {path_str}, total count = {best['count']}"
                )


# ---------- Dual-tightened upper bound (ICP-style) ----------

bound_trace: List[Tuple[int, float, float]] = []  # (depth, tight_bound, naive_bound)

def compute_tight_upper_bound(
    graph: nx.Graph,
    costs: CostMap,
    cut_edges: CutMap,
    max_cycle_length: int = 6
) -> float:
    """
    Dual-feasible tightening via iterative packing of conflicted cycles.
    A conflicted cycle = one negative edge + a positive path between its endpoints.

    We maintain residual capacities cap[e] = |costs[e]| for both signs.
    For each cycle, increase its dual y by min cap on the cycle; subtract from caps.
    Removing depleted positive edges from G_plus prevents reuse. This is a valid
    dual ascent; reduce_total = sum y never exceeds any edge capacity.

    Returned bound:
      naive_positive_sum - reduce_total, clamped at 0.
    """
    # Edges alive in the current node (and not fixed to CUT=1)
    graph_edges = {(min(u, v), max(u, v)) for (u, v) in graph.edges()}
    undecided = {e for e in graph_edges if cut_edges.get(e, -1) != 1}

    E_plus  = {e for e in undecided if costs.get(e, 0.0) > 0.0}
    E_minus = {e for e in undecided if costs.get(e, 0.0) < 0.0}

    # Residual capacities in the dual
    cap: Dict[Edge, float] = {e: abs(costs[e]) for e in (E_plus | E_minus)}

    # Positive-edge subgraph with capacity>0
    def build_G_plus():
        pos_edges_alive = [(u, v) for (u, v) in E_plus if cap.get((min(u, v), max(u, v)), 0.0) > 0.0]
        return graph.edge_subgraph(pos_edges_alive).copy()

    G_plus = build_G_plus()
    reduce_total = 0.0

    improved = True
    while improved:
        improved = False

        # Try to pack cycles per negative edge
        for (u, v) in list(E_minus):
            rep = (min(u, v), max(u, v))
            if cap.get(rep, 0.0) <= 0.0:
                continue
            if (u not in G_plus) or (v not in G_plus):
                continue

            # Shortest positive path (hop distance) u→v
            try:
                path = nx.shortest_path(G_plus, source=u, target=v)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            plen = len(path) - 1
            if plen <= 0 or plen + 1 > max_cycle_length:
                continue

            # Collect positive path edges (must have residual capacity)
            cycle: List[Edge] = []
            feasible = True
            for i in range(plen):
                a, b = path[i], path[i + 1]
                ekey = (min(a, b), max(a, b))
                if ekey not in E_plus or cap.get(ekey, 0.0) <= 0.0:
                    feasible = False
                    break
                cycle.append(ekey)
            if not feasible:
                continue

            # Add the negative edge to complete the cycle
            cycle.append(rep)

            # Dual step
            y = min(cap[e] for e in cycle)
            if y <= 0.0:
                continue

            for e in cycle:
                cap[e] -= y
                if e in E_plus and cap[e] <= 0.0:
                    a, b = e
                    if G_plus.has_edge(a, b):
                        G_plus.remove_edge(a, b)

            reduce_total += y
            improved = True

    naive_upper_bound = sum(costs[e] for e in E_plus)
    bound = max(0.0, naive_upper_bound - reduce_total)
    return bound


# ---------- Utilities for visualization and debug ----------

def plot_bound_trace():
    """Plot tight vs naive bound along the explored depths (then clear the trace)."""
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
    bound_trace.clear()


def print_edge_label_groups(cut_edges: CutMap, tag: str = ""):
    """Grouped dump of labeled edge sets (cut / uncut / undecided)."""
    RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
    cut = {e for e, v in cut_edges.items() if v == 1}
    uncut = {e for e, v in cut_edges.items() if v == 0}
    undecided = {e for e, v in cut_edges.items() if v == -1}

    print(f"[EDGE LABELS{f' @ {tag}' if tag else ''}]")
    print(f"  cut edges:      {RED}{cut}{RESET}")
    print(f"  uncut edges:    {GREEN}{uncut}{RESET}")
    print(f"  undecided edges:{undecided}")


# ---------- Branch-and-Bound ----------

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
    path: Optional[List[str]] = None
):
    """
    Depth-first BnB. Branch on the current highest-cost edge:
      JOIN: contract endpoints (label 0), add its cost to obj.
      CUT:  remove the edge (label 1), obj unchanged.
    Prune if obj + UB < incumbent.
    """
    if node_counter is not None:
        node_counter['count'] += 1

    if path is None:
        path = ["Start"]

    # Leaf: no costs to consider → finalize by cutting undecided edges
    if not costs:
        if log:
            print(f"[NO COSTS] \033[92mcluster_obj={obj:.2f}\033[0m, best_obj={best.get('obj', 0.0):.2f}")
        cut_edges_copy = {k: (1 if v == -1 else v) for k, v in cut_edges.items()}
        update_best_if_feasible_final(orig_graph, cut_edges_copy, obj, best, True, path)
        return None

    # Upper bound: tight (ICP-style) or naive
    if use_tight_bound:
        bound = compute_tight_upper_bound(graph, costs, cut_edges)
    else:
        graph_edges = set(graph.edges())
        bound = sum(
            w for e, w in costs.items()
            if w > 0 and e in graph_edges and cut_edges.get(e, -1) != 1
        )

    if log:
        print(f"[ENTER BnB] \033[92mcluster_obj={obj:.2f}\033[0m, "
              f"\033[91mbound={bound:.2f}\033[0m, best_obj={best.get('obj', 0.0):.2f}")
        naive = sum(
            w for e, w in costs.items()
            if w > 0 and e in graph.edges and cut_edges.get(e, -1) != 1
        )
        bound_trace.append((depth, bound, naive))
        print(f"\033[93m[BOUND] tighter = {bound:.2f}, naive = {naive:.2f}, Δ = {naive - bound:.2f}\033[0m")

    if obj + bound < best.get('obj', 0.0):
        if log:
            print(f"[PRUNE] Max possible obj = {obj + bound:.2f} < best obj = {best.get('obj', 0.0):.2f}")
        return None

    # Pick edge with maximum current cost (ties arbitrary)
    edge, max_cost = max(costs.items(), key=lambda item: item[1])
    u, v = edge
    edge_key = (min(u, v), max(u, v))

    # JOIN branch (only when gain is non-negative)
    if log:
        print_edge_label_groups(cut_edges, f"before JOIN ({u},{v})")
    skip_join = max_cost < 0
    if not skip_join:
        graph_join, costs_join, cut_edges_join = contract_and_merge_costs(graph, costs, u, v, cut_edges, log=log)
        cut_edges_join[edge_key] = 0
        cut_edges_join = propagate_zero_labels(cut_edges_join, u, v, costs, log)
        obj_join = obj + max_cost
        if log:
            print(f"[BRANCH] Join: merging ({u}, {v}) with cost {max_cost:.2f}")
            print(f"  - New objective: {obj_join:.2f}")
        bnb_multicut(
            graph_join, costs_join, cut_edges_join, obj_join, best, log,
            use_tight_bound, depth + 1, node_counter, orig_graph,
            path + [f"JOIN({u},{v})"]
        )
        if log:
            print_edge_label_groups(cut_edges_join, f"after JOIN ({u},{v})")

    # CUT branch
    if log:
        print_edge_label_groups(cut_edges, f"before CUT ({u},{v})")
    graph_cut = graph.copy()
    graph_cut.remove_edge(u, v)
    costs_cut = {e: w for e, w in costs.items() if e != edge}
    cut_edges_cut = cut_edges.copy()
    cut_edges_cut[edge_key] = 1
    if log:
        print(f"[BRANCH] Cut: removing edge {edge_key} with cost {max_cost:.2f}, Objective unchanged: {obj:.2f}")
    bnb_multicut(
        graph_cut, costs_cut, cut_edges_cut, obj, best, log,
        use_tight_bound, depth + 1, node_counter, orig_graph,
        path + [f"CUT({u},{v})"]
    )
    if log:
        print_edge_label_groups(cut_edges_cut, f"after CUT ({u},{v})")


# ---------- Bench + solver wrapper ----------

def benchmark_solver(graph: nx.Graph, costs: CostMap, log: bool = False):
    """
    Compare naive-vs-tight upper bounds on the same instance.
    Reports objective, runtime, and explored nodes; plots bound traces.
    """
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
    """Thin wrapper for a single run of the BnB with a chosen upper bound."""
    def __init__(self, graph: nx.Graph, costs: CostMap, log: bool = True, use_tight_bound: bool = True):
        self.graph = graph
        self.costs = costs
        self.log = log
        self.use_tight_bound = use_tight_bound

    def solve(self):
        if self.log:
            print("graph for bnb solver:")
        # Normalize edge keys (u < v)
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
            print(f"[FINISH] total time = {end - start:.2f} seconds")
            print(f"[STATS] Total nodes visited in BnB: {node_counter['count']}")
            print(f"[BEST PATH] {' -> '.join(best.get('path', []))}")

        # Recompute objective from incumbent labels (safety)
        obj = 0.0
        for u, v in self.graph.edges():
            e = (min(u, v), max(u, v))
            if best['cut'].get(e, -1) == 1:
                obj += normalized_costs.get(e, 0.0)

        if self.log:
            print("[DEBUG] Final raw best cut:", best['cut'])
        return best['cut'], obj, best['count']