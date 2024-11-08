import random
import math
import networkx as nx
from src.solver.sa.simulated_annealing_base_tsp_solver import SimulatedAnnealingBaseTspSolver

class LogarithmicCoolingTwoOptSimulatedAnnealingSolver(SimulatedAnnealingBaseTspSolver):
  """
  Simulated Annealing TSP solver with random initial solution, 2-opt move, and logarithmic cooling.
  """

  def initial_solution(self, graph: nx.Graph):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    return nodes

  def neighbor(self, solution):
    i, j = sorted(random.sample(range(len(solution)), 2))
    new_solution = solution[:i] + solution[i:j+1][::-1] + solution[j+1:]
    return new_solution

  def cool(self, temp):
    """
    Logarithmic cooling schedule.
    """
    return temp / (1 + self.cooling_rate * math.log(1 + temp))
