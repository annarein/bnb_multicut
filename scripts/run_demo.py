from bnb_solver import BnBSolver
from ilp_solver import ILPSolver
from graph_generators import get_random_costs_graph
from visualizer import visualize_multicut_solution, plot_multicut_result
import time

def run_demo():
    graph, costs, pos = get_random_costs_graph(seed=42, shape=(8, 3))
    plot_multicut_result(graph, costs, pos, title="Original Graph")

    ilp = ILPSolver(graph.copy(), costs)
    multicut_ilp, obj_ilp = ilp.solve()
    print(f"[ILP] obj = {obj_ilp}")
    visualize_multicut_solution(graph, costs, pos, multicut_ilp, "ILP Multicut")

    bnb = BnBSolver(graph.copy(), costs)
    multicut_bnb, obj_bnb, count = bnb.solve()
    print(f"[BnB] obj = {obj_bnb}, nodes = {count}")
    visualize_multicut_solution(graph, costs, pos, multicut_bnb, "BnB Multicut")

if __name__ == "__main__":
    run_demo()
