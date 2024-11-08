import random
import networkx as nx
from src.solver.sa.simulated_annealing_base_tsp_solver import SimulatedAnnealingBaseTspSolver

class NearestNeighborTwoOptSimulatedAnnealingSolver(SimulatedAnnealingBaseTspSolver):
  """
  Simulated Annealing TSP solver with nearest neighbor initial solution and 2-opt move.
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
    i, j = sorted(random.sample(range(len(solution)), 2))
    new_solution = solution[:i] + solution[i:j+1][::-1] + solution[j+1:]
    return new_solution
