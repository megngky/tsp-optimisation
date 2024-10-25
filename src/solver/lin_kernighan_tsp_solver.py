import networkx as nx
import numpy as np
import random
import networkx as nx
from src.solver.tsp_solver import TspSolver

class LinKernighanSolver(TspSolver):
  """
  Lin-Kernighan Heuristic for solving the TSP.
  This heuristic dynamically performs k-opt moves to iteratively improve a TSP tour.
  """

  def solve(self, graph: nx.Graph):
    """
    Solve the TSP for a given graph using the Lin-Kernighan Heuristic.

    Parameters:
    graph (networkx.Graph): The graph representing the TSP problem.

    Returns:
    tuple: (tour, total_cost) 
    """
    # Generate an initial random tour
    nodes = list(graph.nodes)
    current_tour = nodes.copy()
    random.shuffle(current_tour)

    # Calculate initial tour cost
    best_tour = current_tour
    best_cost = self.calculate_tour_cost(graph, current_tour)
    
    # Start local search using k-opt moves
    improved = True
    while improved:
      improved = False
      for i in range(len(current_tour) - 1):
        for j in range(i + 1, len(current_tour)):
          new_tour = self.perform_2opt_swap(current_tour, i, j)
          new_cost = self.calculate_tour_cost(graph, new_tour)
          
          # Check if the new tour is better
          if new_cost < best_cost:
            best_tour = new_tour
            best_cost = new_cost
            improved = True

      current_tour = best_tour

    return best_tour, best_cost

  def calculate_tour_cost(self, graph, tour):
    """
    Calculate the total cost of the tour.

    Parameters:
    graph (networkx.Graph): The graph representing the TSP problem.
    tour (list): The list of nodes representing the current tour.

    Returns:
    float: The total cost of the tour.
    """
    total_cost = 0
    for i in range(len(tour) - 1):
      total_cost += graph[tour[i]][tour[i+1]]['weight']
    total_cost += graph[tour[-1]][tour[0]]['weight']  # Complete the tour
    return total_cost

  def perform_2opt_swap(self, tour, i, j):
    """
    Perform a 2-opt swap by reversing the order of the cities between index i and j.

    Parameters:
    tour (list): The list of nodes representing the current tour.
    i (int): The start index of the swap.
    j (int): The end index of the swap.

    Returns:
    list: The new tour after performing the 2-opt swap.
    """
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour
