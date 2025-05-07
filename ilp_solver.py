import gurobipy as gp
from gurobipy import GRB
import networkx as nx

class ILPSolver:
    def __init__(self, graph: nx.Graph, costs: dict, log: bool = False):
        self.graph = graph
        self.costs = costs
        self.log = log
        self.model = None
        self.variables = None

    def _separate_cycle_inequalities(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return
        vals = model.cbGetSolution(self.variables)
        g_copy = self.graph.copy()
        g_copy.remove_edges_from([e for e in g_copy.edges if vals.get((min(e), max(e)), 0) > 0.5])
        components = nx.connected_components(g_copy)
        node_labeling = {n: i for i, comp in enumerate(components) for n in comp}
        for (u, v), x_uv in vals.items():
            if x_uv < 0.5 or node_labeling[u] != node_labeling[v]:
                continue
            path = nx.shortest_path(g_copy, u, v)
            assert len(path) >= 2
            model.cbLazy(self.variables[u, v] <= gp.quicksum(
                self.variables[min(path[i], path[i + 1]), max(path[i], path[i + 1])]
                for i in range(len(path) - 1)
            ))

    def solve(self):
        self.model = gp.Model()
        self.model.setParam('OutputFlag', 1 if self.log else 0)
        self.model.Params.LazyConstraints = 1
        self.variables = self.model.addVars(self.costs.keys(), obj=self.costs, vtype=GRB.BINARY, name='e')
        self.model.optimize(self._separate_cycle_inequalities)

        solution = self.model.getAttr("X", self.variables)
        multicut = {(min(u, v), max(u, v)): 1 if x_e > 0.5 else 0 for (u, v), x_e in solution.items()}
        return multicut, self.model.ObjVal