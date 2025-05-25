from bnb_solver import BnBSolver
from cp_loader import load_cp_lib_instance
from evaluator import extract_opt_value, parse_opt_solution
from visualizer import visualize_multicut_solution, plot_multicut_result
import networkx as nx
import os
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
INSTANCE_PATH = os.path.join(parent_dir, "cp_lib/ABR/cars.txt")
VISUALIZE = True


def multicut_to_clusters(graph, cut_edges):
    g_copy = graph.copy()
    for u, v in list(g_copy.edges):
        if cut_edges.get((min(u, v), max(u, v)), 0) == 1:
            g_copy.remove_edge(u, v)
    clusters = list(nx.connected_components(g_copy))
    return clusters


def print_clusters(title, clusters):
    print(title)
    for cluster in clusters:
        cluster_str = " ".join(str(n + 1) for n in sorted(cluster))  # 输出 1-based
        print("{ " + cluster_str + " }")


def run_cp_instance():
    graph, costs = load_cp_lib_instance(INSTANCE_PATH)

    bnb = BnBSolver(graph.copy(), costs)
    multicut, obj_bnb, count = bnb.solve()
    print(f"[BnB] obj = {obj_bnb}, nodes = {count}")

    clusters_bnb = multicut_to_clusters(graph, multicut)
    print_clusters("Clusters (BnB):", clusters_bnb)

    pos = nx.spring_layout(graph, seed=42)
    if VISUALIZE:
        visualize_multicut_solution(graph, costs, pos, multicut, "BnB Multicut")

    opt_path = INSTANCE_PATH.replace(".txt", "_opt.txt").replace("ABR/", "ABR/Optimal/")
    if os.path.exists(opt_path):
        known_val = extract_opt_value(opt_path)
        total_weight = sum(costs.values())
        expected = total_weight - known_val
        print(f"[OPT] known = {known_val}, expected BnB = {expected}")
        print(f"[MATCH] ✅ {abs(obj_bnb - expected) < 1e-6}")

        clusters_opt = parse_opt_solution(opt_path)
        clusters_opt = [set(node - 1 for node in group) for group in clusters_opt]
        print_clusters("Clusters (Optimal):", clusters_opt)

        if VISUALIZE:
            node_to_cluster = {node: i for i, group in enumerate(clusters_opt) for node in group}
            plot_multicut_result(graph, costs, pos, node_labeling=node_to_cluster, title="Optimal Partition")
    else:
        print("⚠️ Optimal file not found.")


if __name__ == "__main__":
    run_cp_instance()
