import random
import numpy as np
import networkx as nx
from src.solver.tsp_solver import TspSolver

class GeneticAlgorithmSolver(TspSolver):
  def __init__(self, population_size=100, mutation_rate=0.01, generations=500):
    self.population_size = population_size
    self.mutation_rate = mutation_rate
    self.generations = generations

  def solve(self, graph: nx.Graph):
    nodes = list(graph.nodes)
    population = self._initialize_population(nodes)

    for generation in range(self.generations):
      fitness_scores = self._evaluate_population(population, graph)
      population = self._select_next_generation(population, fitness_scores)
      population = self._mutate_population(population)

    # Get the best individual (shortest tour)
    best_individual = min(population, key=lambda p: self._calculate_total_cost(p, graph))
    total_cost = self._calculate_total_cost(best_individual, graph)

    return best_individual, total_cost

  def _initialize_population(self, nodes):
    population = [random.sample(nodes, len(nodes)) for _ in range(self.population_size)]
    return population

  def _evaluate_population(self, population, graph):
    return [1 / self._calculate_total_cost(individual, graph) for individual in population]

  def _calculate_total_cost(self, tour, graph):
    return sum(graph[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1)) + graph[tour[-1]][tour[0]]['weight']

  def _select_next_generation(self, population, fitness_scores):
    probabilities = np.array(fitness_scores) / sum(fitness_scores)
    selected = random.choices(population, weights=probabilities, k=self.population_size)
    next_generation = [self._crossover(random.choice(selected), random.choice(selected)) for _ in range(self.population_size)]
    return next_generation

  def _crossover(self, parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    ptr = end

    for node in parent2:
      if node not in child:
        if ptr >= size:
          ptr = 0
        child[ptr] = node
        ptr += 1

    return child

  def _mutate_population(self, population):
    for individual in population:
      if random.random() < self.mutation_rate:
        self._mutate(individual)
    return population

  def _mutate(self, individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
