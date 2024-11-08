import random
from src.solver.ga.genetic_base_tsp_solver import GeneticAlgorithmBaseTspSolver

class TournamentSelectionGeneticAlgorithmSolver(GeneticAlgorithmBaseTspSolver):
  def _select_next_generation(self, population, fitness_scores):
    # Apply elitism by carrying over top individuals
    elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_count]
    next_generation = [population[i] for i in elite_indices]

    # Tournament selection for remaining individuals
    while len(next_generation) < self.population_size:
      parent1 = self._tournament_selection(population, fitness_scores)
      parent2 = self._tournament_selection(population, fitness_scores)
      child = self._crossover(parent1, parent2)

      # Apply 2-opt optimization on offspring
      optimized_child = self._two_opt(child)
      next_generation.append(optimized_child)

    return next_generation

  def _tournament_selection(self, population, fitness_scores, tournament_size=5):
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = max(selected_indices, key=lambda i: fitness_scores[i])
    return population[best_index]

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
