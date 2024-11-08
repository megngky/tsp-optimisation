import random
import math
import networkx as nx
from abc import ABC, abstractmethod
from src.solver.tsp_solver import TspSolver

class SimulatedAnnealingBaseTspSolver(TspSolver, ABC):
  """
  Abstract base class for Simulated Annealing TSP solvers, defining common structure.
  """

  def __init__(self, initial_temp=1000, cooling_rate=0.995, stopping_temp=1):
    self.initial_temp = initial_temp
    self.cooling_rate = cooling_rate
    self.stopping_temp = stopping_temp

  def solve(self, graph: nx.Graph):
    """
    Solve the TSP using Simulated Annealing.

    Parameters:
    graph (networkx.Graph): The graph representing the TSP problem.
    
    Returns:
    tuple: (best_solution, best_cost) 
    """
    # Store graph as an instance attribute
    self.graph = graph

    # Start with an initial solution
    current_solution = self.initial_solution(graph)
    current_cost = self.calculate_total_cost(current_solution, graph)
    
    best_solution = current_solution[:]
    best_cost = current_cost
    temp = self.initial_temp

    # Simulated Annealing loop
    while temp > self.stopping_temp:
      new_solution = self.neighbor(current_solution)
      new_cost = self.calculate_total_cost(new_solution, graph)

      # Acceptance probability (Metropolis criterion)
      acceptance_probability = math.exp((current_cost - new_cost) / temp) if new_cost > current_cost else 1
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

  def calculate_total_cost(self, solution, graph):
    """
    Calculate the total cost of a given solution.

    Parameters:
    solution (list): A list representing a tour of nodes.
    graph (networkx.Graph): The graph representing the TSP problem.

    Returns:
    float: The total cost of the tour.
    """
    total_cost = 0
    for i in range(len(solution) - 1):
      total_cost += graph[solution[i]][solution[i + 1]]['weight']
    total_cost += graph[solution[-1]][solution[0]]['weight']
    return total_cost

  @abstractmethod
  def initial_solution(self, graph: nx.Graph):
    """Generate an initial solution."""
    pass

  @abstractmethod
  def neighbor(self, solution):
    """Generate a neighboring solution."""
    pass
