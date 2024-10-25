import random
import math
import networkx as nx
from src.solver.tsp_solver import TspSolver

class SimulatedAnnealingTspSolver(TspSolver):
  """
  Simulated Annealing algorithm-based solver for the Traveling Salesman Problem (TSP).
  """

  def __init__(self, initial_temp=1000, cooling_rate=0.995, stopping_temp=1):
    self.initial_temp = initial_temp  # Initial temperature
    self.cooling_rate = cooling_rate  # Cooling rate
    self.stopping_temp = stopping_temp  # Minimum temperature to stop

  def solve(self, graph: nx.Graph):
    nodes = list(graph.nodes)
    
    # Initial solution (random tour)
    current_solution = nodes.copy()
    random.shuffle(current_solution)
    
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
      # Swap two nodes to create a new solution
      new_solution = current_solution[:]
      i, j = random.sample(range(len(new_solution)), 2)
      new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

      # Calculate cost of the new solution
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
