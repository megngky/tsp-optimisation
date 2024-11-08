import random
import networkx as nx
from src.solver.sa.simulated_annealing_base_tsp_solver import SimulatedAnnealingBaseTspSolver

class RandomSwapSimulatedAnnealingSolver(SimulatedAnnealingBaseTspSolver):
  """
  Simulated Annealing TSP solver with random initial solution and simple swap move.
  """

  def initial_solution(self, graph: nx.Graph):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    return nodes

  def neighbor(self, solution):
    new_solution = solution[:]
    i, j = random.sample(range(len(new_solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution
