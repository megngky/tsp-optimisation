import random
import networkx as nx
from src.solver.sa.simulated_annealing_base_tsp_solver import SimulatedAnnealingBaseTspSolver

class GreedySwapSimulatedAnnealingSolver(SimulatedAnnealingBaseTspSolver):
  """
  Simulated Annealing TSP solver with greedy initial solution and simple swap move.
  """

  def initial_solution(self, graph: nx.Graph):
    nodes = list(graph.nodes)
    unvisited = set(nodes)
    start_node = random.choice(nodes)
    tour = [start_node]
    unvisited.remove(start_node)

    while unvisited:
      last_node = tour[-1]
      next_node = min(unvisited, key=lambda node: graph[last_node][node]['weight'])
      tour.append(next_node)
      unvisited.remove(next_node)

    return tour

  def neighbor(self, solution):
    new_solution = solution[:]
    i, j = random.sample(range(len(new_solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution
