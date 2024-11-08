import random
import networkx as nx
from abc import ABC, abstractmethod
from multiprocessing import Pool
from src.solver.tsp_solver import TspSolver

class GeneticAlgorithmBaseTspSolver(TspSolver, ABC):
  def __init__(self, population_size=20, mutation_rate=0.05, generations=100, elite_count=2, early_stop_rounds=10):
    self.population_size = population_size
    self.mutation_rate = mutation_rate
    self.generations = generations
    self.elite_count = elite_count
    self.early_stop_rounds = early_stop_rounds

  def solve(self, graph: nx.Graph):
    # Early exit if the number of nodes is too large for GA to handle efficiently
    if len(graph.nodes) > 50:
      return None  # Not suitable for large TSP instances

    # Set graph as an instance attribute
    self.graph = graph
    nodes = list(graph.nodes)
    population = self._initialize_population(nodes)
    best_cost = float('inf')
    no_improvement_rounds = 0

    for generation in range(self.generations):
      fitness_scores = self._evaluate_population(population, graph)
      population = self._select_next_generation(population, fitness_scores)
      population = self._mutate_population(population)

      # Apply 2-opt only on the top-performing individuals
      population = self._apply_two_opt_to_elite(population, fitness_scores)

      # Track best solution and apply early stopping
      current_best = min(population, key=lambda p: self._calculate_total_cost(p, graph))
      current_best_cost = self._calculate_total_cost(current_best, graph)

      if current_best_cost < best_cost:
        best_cost = current_best_cost
        best_individual = current_best
        no_improvement_rounds = 0
      else:
        no_improvement_rounds += 1
        if no_improvement_rounds >= self.early_stop_rounds:
          break  # Stop if no improvement for early_stop_rounds generations

      # Optionally adjust mutation rate adaptively
      self.mutation_rate = self._adjust_mutation_rate(generation)

    return best_individual, best_cost

  def _initialize_population(self, nodes):
    population = [self._nearest_neighbor_solution(nodes) for _ in range(self.population_size // 5)]
    population += [random.sample(nodes, len(nodes)) for _ in range(self.population_size - len(population))]
    return population

  def _nearest_neighbor_solution(self, nodes):
    start = random.choice(nodes)
    tour = [start]
    unvisited = set(nodes)
    unvisited.remove(start)
    current = start

    while unvisited:
      next_node = min(unvisited, key=lambda node: self.graph[current][node]['weight'])
      tour.append(next_node)
      unvisited.remove(next_node)
      current = next_node

    return tour

  @abstractmethod
  def _select_next_generation(self, population, fitness_scores):
    pass

  @abstractmethod
  def _crossover(self, parent1, parent2):
    pass

  def _mutate_population(self, population):
    for individual in population:
      if random.random() < self.mutation_rate:
        self._mutate(individual)
    return population

  def _mutate(self, individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]

  def _adjust_mutation_rate(self, generation):
    return self.mutation_rate  # By default, it remains constant

  def _evaluate_population(self, population, graph):
    with Pool() as pool:
      costs = pool.starmap(self._calculate_total_cost, [(individual, graph) for individual in population])
    return [1 / cost for cost in costs]

  def _calculate_total_cost(self, tour, graph):
    return sum(graph[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1)) + graph[tour[-1]][tour[0]]['weight']

  def _apply_two_opt_to_elite(self, population, fitness_scores):
    """Apply 2-opt local search only to the top-performing individuals."""
    elite_count = max(1, self.population_size // 5)  # Apply to top 20% of population
    elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
    for idx in elite_indices:
      population[idx] = self._two_opt(population[idx])
    return population

  def _two_opt(self, tour):
    """Apply 2-opt local search to optimize the tour."""
    best = tour
    improved = True
    while improved:
      improved = False
      for i in range(1, len(tour) - 2):
        for j in range(i + 1, len(tour)):
          if j - i == 1: continue  # Skip adjacent edges
          new_tour = tour[:]
          new_tour[i:j] = tour[j - 1:i - 1:-1]  # Reverse the tour segment
          if self._calculate_total_cost(new_tour, self.graph) < self._calculate_total_cost(best, self.graph):
            best = new_tour
            improved = True
      tour = best
    return best
