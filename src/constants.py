DIR_DATA = "data"
DIR_GRAPH = "graph"
DIR_PICKLE = "pickle"

FILE_DATA_FRAME = f"{DIR_PICKLE}/tsp_problems_dataframe.pkl"
FILE_OPTIMAL_COSTS = f"{DIR_DATA}/optimal_costs.json"

TIME_TAKEN_PLACEHOLDER = "Time taken to {action}: {length:.5f}s"

SOLUTION_DP = "dp"
SOLUTION_GREEDY = "greedy"
SOLUTION_CHRISTOFIDES = "christofides"
SOLUTION_SIMULATED_ANNEALING = "sa"
SOLUTION_LIN_KERNIGHAN = "lk"
SOLUTION_GENETIC = "genetic"
SOLUTION_ANT_COLONY = "ant"
ALL_SOLUTIONS = [
  SOLUTION_DP,
  SOLUTION_GREEDY,
  SOLUTION_CHRISTOFIDES,
  SOLUTION_SIMULATED_ANNEALING
  # SOLUTION_LIN_KERNIGHAN,
  # SOLUTION_GENETIC,
  # SOLUTION_ANT_COLONY
]
