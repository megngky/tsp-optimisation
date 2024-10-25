import random
import numpy as np
import networkx as nx
from src.solver.tsp_solver import TspSolver

class AntColonySolver(TspSolver):
  def __init__(self, num_ants=50, num_iterations=100, alpha=1, beta=2, evaporation_rate=0.5, pheromone_init=0.1):
    self.num_ants = num_ants
    self.num_iterations = num_iterations
    self.alpha = alpha  # influence of pheromone
    self.beta = beta    # influence of heuristic information
    self.evaporation_rate = evaporation_rate
    self.pheromone_init = pheromone_init

  def solve(self, graph: nx.Graph):
    nodes = list(graph.nodes)
    num_nodes = len(nodes)
    pheromone_matrix = np.full((num_nodes, num_nodes), self.pheromone_init)

    best_tour = None
    best_cost = float('inf')

    for _ in range(self.num_iterations):
      all_tours = []
      all_costs = []

      for ant in range(self.num_ants):
        tour = self._construct_solution(graph, nodes, pheromone_matrix)
        cost = self._calculate_total_cost(tour, graph)

        all_tours.append(tour)
        all_costs.append(cost)

        if cost < best_cost:
          best_tour = tour
          best_cost = cost

      # Update pheromone levels
      self._evaporate_pheromone(pheromone_matrix)
      self._update_pheromones(pheromone_matrix, all_tours, all_costs, graph)

    return best_tour, best_cost

  def _construct_solution(self, graph, nodes, pheromone_matrix):
    tour = [random.choice(nodes)]
    unvisited = set(nodes) - {tour[0]}

    while unvisited:
      current_node = tour[-1]
      probabilities = self._calculate_probabilities(current_node, unvisited, pheromone_matrix, graph)
      next_node = random.choices(list(unvisited), weights=probabilities)[0]
      tour.append(next_node)
      unvisited.remove(next_node)

    return tour

  def _calculate_probabilities(self, current_node, unvisited, pheromone_matrix, graph):
    current_idx = current_node
    probabilities = []

    for next_node in unvisited:
      next_idx = next_node
      pheromone = pheromone_matrix[current_idx][next_idx]
      distance = graph[current_node][next_node]['weight']
      heuristic_value = 1 / distance if distance > 0 else 1
      probabilities.append((pheromone ** self.alpha) * (heuristic_value ** self.beta))

    return np.array(probabilities) / sum(probabilities)

  def _calculate_total_cost(self, tour, graph):
    return sum(graph[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1)) + graph[tour[-1]][tour[0]]['weight']

  def _evaporate_pheromone(self, pheromone_matrix):
    pheromone_matrix *= (1 - self.evaporation_rate)

  def _update_pheromones(self, pheromone_matrix, all_tours, all_costs, graph):
    for tour, cost in zip(all_tours, all_costs):
      for i in range(len(tour) - 1):
        pheromone_matrix[tour[i]][tour[i + 1]] += 1 / cost
      pheromone_matrix[tour[-1]][tour[0]] += 1 / cost
