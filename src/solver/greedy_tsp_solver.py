import networkx as nx
from src.solver.tsp_solver import TspSolver

class GreedyTspSolver(TspSolver):
	"""
	Greedy algorithm-based solver for the Traveling Salesman Problem (TSP).
	"""

	def solve(self, graph: nx.Graph):
		start_node = list(graph.nodes)[0]  # Start from the first node
		visited = [start_node]
		total_cost = 0
		current_node = start_node

		while len(visited) < len(graph.nodes):
			neighbors = [(neighbor, graph[current_node][neighbor]['weight'])
						 for neighbor in graph.neighbors(current_node)
						 if neighbor not in visited]
			
			nearest_neighbor, cost = min(neighbors, key=lambda x: x[1])
			visited.append(nearest_neighbor)
			total_cost += cost
			current_node = nearest_neighbor
		
		return_to_start_cost = graph[current_node][start_node]['weight']
		total_cost += return_to_start_cost
		visited.append(start_node)

		return visited, total_cost
