import random
import numpy as np
from src.solver.ga.genetic_base_tsp_solver import GeneticAlgorithmBaseTspSolver

class AdaptiveMutationGeneticAlgorithmSolver(GeneticAlgorithmBaseTspSolver):
  def _select_next_generation(self, population, fitness_scores):
    probabilities = np.array(fitness_scores) / sum(fitness_scores)
    selected = random.choices(population, weights=probabilities, k=self.population_size)
    
    # Apply crossover and local search (2-opt) on each child
    next_generation = [self._two_opt(self._crossover(random.choice(selected), random.choice(selected)))
                       for _ in range(self.population_size)]
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

  def _adjust_mutation_rate(self, generation):
    # Adaptive mutation rate: decrease over generations
    return max(0.01, self.mutation_rate * (0.995 ** generation))
