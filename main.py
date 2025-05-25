import time
from ilp_solver import ILPSolver
from bnb_solver import BnBSolver, benchmark_solver
from graph_generators import get_random_costs_graph, get_trivial_graph, get_test_zeros_graph
import networkx as nx
from visualizer import visualize_multicut_solution, plot_multicut_result
import time


def main():
    graph, costs, pos = get_random_costs_graph(seed=37, shape=(5, 8))
    for u, v in graph.edges():
        print(u, v, costs[(u, v)])

    # Original graph visualization (optional)
    # plot_multicut_result(graph, costs, pos, multicut=None, node_labeling=None, title="Original Graph")


    # === ILP Solver ===
    solver_ilp = ILPSolver(graph.copy(), costs)
    start_time = time.time()
    multicut_ilp, obj_ilp = solver_ilp.solve()
    elapsed_ilp = time.time() - start_time
    print(f"ILP_multicut took {elapsed_ilp:.4f} seconds")
    visualize_multicut_solution(graph, costs, pos, multicut_ilp, "ILP Multicut Result")

    # === Branch and Bound: benchmark both naive & tight ===
    benchmark_solver(graph, costs, log=True)  # Turn off detailed logging for clean output

    # === Branch and Bound Solver ===
    solver_bnb = BnBSolver(graph.copy(), costs, False)
    start_time = time.time()
    multicut_bnb, obj_bnb, count_bnb = solver_bnb.solve()
    elapsed_bnb = time.time() - start_time
    print(f"bnb_multicut took {elapsed_bnb:.4f} seconds")
    print(f"count_bnb: {count_bnb}")
    # print(obj_bnb, obj_ilp)
    visualize_multicut_solution(graph, costs, pos, multicut_bnb, "BnB Multicut Result")


def benchmark(num_instances=1000, shape=(2, 3), tolerance=1e-6):
    for seed in range(num_instances):
        graph, costs, pos = get_random_costs_graph(seed=seed, shape=shape)
        # graph, costs, pos = get_test_zeros_graph()
        # graph, costs, pos = get_trivial_graph()

        # plot_multicut_result(graph, costs, pos, title="Original Graph")

        # === ILP Solver ===
        solver_ilp = ILPSolver(graph.copy(), costs)
        start_time = time.time()
        multicut_ilp, obj_ilp = solver_ilp.solve()
        elapsed_ilp = time.time() - start_time
        print(f"ILP_multicut took {elapsed_ilp:.4f} seconds")
        #
        # node_labeling_ilp = get_node_labeling(graph, multicut_ilp)
        # plot_multicut_result(graph, costs, pos, multicut_ilp, node_labeling_ilp, title="ILP Multicut Result")

        # === Branch and Bound Solver ===
        solver_bnb = BnBSolver(graph.copy(), costs)
        start_time = time.time()
        multicut_bnb, obj_bnb, count_bnb = solver_bnb.solve()
        elapsed_bnb = time.time() - start_time
        print(f"bnb_multicut took {elapsed_bnb:.4f} seconds")
        print(f"count_bnb: {count_bnb}")
        print(obj_bnb, obj_ilp)
        print(seed)
        assert abs(obj_bnb - obj_ilp) < tolerance

        # node_labeling_bnb = get_node_labeling(graph, multicut_bnb)
        # plot_multicut_result(graph, costs, pos, multicut_bnb, node_labeling_bnb, title="BnB Multicut Result")


def run_cp_lib_instance():
    from cp_loader import load_cp_lib_instance
    from evaluator import parse_opt_solution, extract_opt_value
    import os

    path = "cp_lib/ABR/workers.txt"
    graph, costs = load_cp_lib_instance(path)

    solver_bnb = BnBSolver(graph.copy(), costs, True)
    multicut_bnb, obj_bnb, count_bnb = solver_bnb.solve()
    print(f"[BnB] obj = {obj_bnb}, nodes = {count_bnb}")

    pos = nx.spring_layout(graph, seed=42)
    visualize_multicut_solution(graph, costs, pos, multicut_bnb, "BnB Multicut Result")

    opt_path = path.replace(".txt", "_opt.txt").replace("ABR/", "ABR/Optimal/")
    if os.path.exists(opt_path):
        known_opt_val = extract_opt_value(opt_path)
        total_weight = sum(costs.values())
        expected_bnb_obj = total_weight - known_opt_val
        print(f"[OPT] known = {known_opt_val}, expected BnB obj = {expected_bnb_obj}")
        print(f"[MATCH] ✅ {abs(obj_bnb - expected_bnb_obj) < 1e-6}")
    else:
        print("⚠️ Optimal solution file not found.")


if __name__ == "__main__":
    main()  # for single test + visualization
    # benchmark(num_instances=100, shape=(2, 3))  # for batch correctness check
    # run_cp_lib_instance()
