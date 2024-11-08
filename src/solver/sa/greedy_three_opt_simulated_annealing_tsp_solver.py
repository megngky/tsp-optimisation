import random
import networkx as nx
from src.solver.sa.simulated_annealing_base_tsp_solver import SimulatedAnnealingBaseTspSolver

class GreedyThreeOptSimulatedAnnealingSolver(SimulatedAnnealingBaseTspSolver):
  """
  Simulated Annealing TSP solver with greedy initial solution and 3-opt move.
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
    """
    3-opt swap: removes three edges and reconnects the segments in all possible ways, 
    choosing the one with the lowest total cost.
    """
    n = len(solution)
    i, j, k = sorted(random.sample(range(n), 3))
    
    # Try different reconnections and choose the best one
    new_solutions = [
      solution[:i] + solution[i:j][::-1] + solution[j:k][::-1] + solution[k:],  # Reverse both segments
      solution[:i] + solution[j:k] + solution[i:j] + solution[k:],              # Swap segments
      solution[:i] + solution[j:k][::-1] + solution[i:j][::-1] + solution[k:],  # Reverse and swap
    ]
    
    # Evaluate and return the best new solution
    best_solution = min(new_solutions, key=lambda sol: self.calculate_total_cost(sol, self.graph))
    return best_solution



