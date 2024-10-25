import networkx as nx
import numpy as np
from src.solver.tsp_solver import TspSolver

class DynamicProgrammingTspSolver(TspSolver):
  """
  TSP solver using the Held-Karp (Dynamic Programming) algorithm.
  """

  def solve(self, graph: nx.Graph):
    n = len(graph.nodes)  # Number of cities/nodes

    if n > 20:
      return None

    all_nodes = list(graph.nodes)
    
    # Distance matrix
    dist = np.zeros((n, n))
    for i, u in enumerate(all_nodes):
      for j, v in enumerate(all_nodes):
        if graph.has_edge(u, v):
          dist[i, j] = graph[u][v]['weight']
        else:
          dist[i, j] = float('inf')  # No edge means infinite cost

    # Held-Karp DP table (stores subproblem solutions)
    # dp[mask][i] -> minimum cost to visit subset 'mask' ending at node 'i'
    dp = np.full((1 << n, n), np.inf)
    dp[1][0] = 0  # Starting at node 0, cost is 0

    # Iterate through subsets of increasing size
    for mask in range(1, 1 << n):
      for u in range(n):
        if not (mask & (1 << u)):
          continue
        for v in range(n):
          if mask & (1 << v) or u == v:
            continue
          dp[mask | (1 << v)][v] = min(
            dp[mask | (1 << v)][v],
            dp[mask][u] + dist[u][v]
          )

    # Reconstruct the tour
    mask = (1 << n) - 1  # All nodes visited
    last_node = 0
    min_cost = np.inf
    for u in range(1, n):
      if dp[mask][u] + dist[u][0] < min_cost:
        min_cost = dp[mask][u] + dist[u][0]
        last_node = u

    tour = [0]  # Start at the first node
    current_mask = mask
    current_node = last_node

    # Backtrack to find the optimal tour
    for _ in range(n - 1):
      tour.append(current_node)
      next_mask = current_mask ^ (1 << current_node)
      next_node = min(
        range(n),
        key=lambda u: dp[next_mask][u] + dist[u][current_node] if current_mask & (1 << u) else np.inf
      )
      current_mask = next_mask
      current_node = next_node

    tour.append(0)  # Complete the tour by returning to the starting point

    return tour, min_cost
