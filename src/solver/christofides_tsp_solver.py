import networkx as nx
from networkx.algorithms.matching import max_weight_matching
from networkx.algorithms.tree import minimum_spanning_tree
from src.solver.tsp_solver import TspSolver

class ChristofidesTspSolver(TspSolver):
  """
  Christofides algorithm-based solver for the Traveling Salesman Problem (TSP).
  This is an approximation algorithm with a guaranteed bound of 1.5 times the optimal solution.
  """

  def solve(self, graph: nx.Graph):
    # Step 1: Find the Minimum Spanning Tree (MST) of the graph
    mst = minimum_spanning_tree(graph)

    # Step 2: Find vertices with an odd degree in the MST
    odd_degree_nodes = [node for node in mst.nodes if mst.degree[node] % 2 != 0]

    # Step 3: Find a minimum-weight perfect matching for the odd-degree vertices
    subgraph_odd = graph.subgraph(odd_degree_nodes)
    matching = max_weight_matching(subgraph_odd, maxcardinality=True)

    # Step 4: Combine MST and the matching to form a multigraph
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(matching)

    # Step 5: Find an Eulerian circuit (every edge is visited exactly once)
    eulerian_circuit = list(nx.eulerian_circuit(multigraph))

    # Step 6: Convert the Eulerian circuit into a Hamiltonian cycle by shortcutting
    visited = []
    tour_cost = 0
    visited_set = set()

    for u, v in eulerian_circuit:
      if u not in visited_set:
        visited.append(u)
        visited_set.add(u)
      tour_cost += graph[u][v]['weight']  # Add the cost of the edge

    # Complete the cycle by returning to the start node
    visited.append(visited[0])
    tour_cost += graph[visited[-2]][visited[0]]['weight']

    return visited, tour_cost
