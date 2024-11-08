import random
import math
import networkx as nx
from src.solver.tsp_solver import TspSolver

class ImprovedSaTspSolver(TspSolver):
  """
  Simulated Annealing algorithm-based solver for the Traveling Salesman Problem (TSP).
  """

  def __init__(self, initial_temp=1000, cooling_rate=0.995, stopping_temp=1):
    self.initial_temp = initial_temp  # Initial temperature
    self.cooling_rate = cooling_rate  # Cooling rate
    self.stopping_temp = stopping_temp  # Minimum temperature to stop

  def solve(self, graph: nx.Graph):
    nodes = list(graph.nodes)

    # Use a nearest neighbor heuristic to create an initial solution
    current_solution = self.nearest_neighbor_heuristic(nodes, graph)
    
    def calculate_total_cost(solution):
      total_cost = 0
      for i in range(len(solution) - 1):
        total_cost += graph[solution[i]][solution[i + 1]]['weight']
      total_cost += graph[solution[-1]][solution[0]]['weight']  # Return to start node
      return total_cost

    # Calculate cost of the initial solution
    current_cost = calculate_total_cost(current_solution)
    best_solution = current_solution[:]
    best_cost = current_cost
    temp = self.initial_temp

    # Simulated Annealing loop
    while temp > self.stopping_temp:
      # Perform a 2-opt move to generate a new solution
      new_solution = self.two_opt_swap(current_solution)
      new_cost = calculate_total_cost(new_solution)

      # Acceptance probability (Metropolis criterion)
      acceptance_probability = math.exp((current_cost - new_cost) / temp) if new_cost > current_cost else 1

      # Accept the new solution with a certain probability
      if random.random() < acceptance_probability:
        current_solution = new_solution
        current_cost = new_cost

        # Update the best solution found so far
        if new_cost < best_cost:
          best_solution = new_solution
          best_cost = new_cost

      # Decrease temperature
      temp *= self.cooling_rate

    # Complete the cycle by returning to the start node
    best_solution.append(best_solution[0])

    return best_solution, best_cost

  def nearest_neighbor_heuristic(self, nodes, graph):
    """Generates an initial solution using the nearest neighbor heuristic."""
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

  def two_opt_swap(self, solution):
    """Performs a 2-opt swap on the solution."""
    i, j = sorted(random.sample(range(len(solution)), 2))
    new_solution = solution[:i] + solution[i:j+1][::-1] + solution[j+1:]
    return new_solution
