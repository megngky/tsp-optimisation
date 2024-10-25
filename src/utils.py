import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET
from src.constants import (
  TIME_TAKEN_PLACEHOLDER,
  DIR_DATA,
  DIR_GRAPH,
  FILE_DATA_FRAME,
  FILE_OPTIMAL_COSTS
)


def print_time_taken(action, start, end):
  length = end - start
  print(TIME_TAKEN_PLACEHOLDER.format(action=action, length=length))


def parse_xml(file_path):
  start = time.time()

  tree = ET.parse(file_path)
  root = tree.getroot()
  
  # Extract graph data
  vertices = []
  for vertex in root.find('graph'):
    edges = []
    for edge in vertex.findall('edge'):
      target = int(edge.text)  # The other vertex
      cost = float(edge.get('cost'))  # The cost of the edge
      edges.append((target, cost))
    vertices.append(edges)
  
  print(f"Number of vertices: {len(vertices)}")
  end = time.time()
  print_time_taken("parse xml", start, end)

  return vertices


def create_graph(vertices):
  start = time.time()

  G = nx.Graph()  # Symmetric TSP, hence undirected graph

  # Iterate over the vertices and add edges
  for i, edges in enumerate(vertices):
    for target, cost in edges:
      G.add_edge(i, target, weight=cost)

  end = time.time()
  print_time_taken("create graph", start, end)
  
  return G


def plot_and_save_graph(G, output_file):
  start = time.time()
  
  # Inform that layout calculation is starting
  print("Calculating layout positions...")
  pos = nx.spring_layout(G)  # Layout for nodes
  print("Layout calculation completed.")
  
  plt.figure()
  
  # Draw the graph
  print("Drawing the graph...")
  nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
  labels = nx.get_edge_attributes(G, 'weight')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
  
  # Save the graph to a file
  print(f"Saving the graph to {output_file}...")
  plt.savefig(output_file)
  plt.close()  # Close the plot to free memory

  end = time.time()
  print_time_taken("plot and save graph", start, end)


def process_all_files_to_png(data_dir=DIR_DATA, output_dir=DIR_GRAPH):
  # Get the total number of files to process
  xml_files = [f for f in os.listdir(data_dir) if f.endswith(".xml")]
  total_files = len(xml_files)
  
  for idx, filename in enumerate(xml_files, 1):
    print(f"Processing {filename[:-4]} [{idx}/{total_files}]...")
    file_path = os.path.join(data_dir, filename)
    
    # Parse XML and create the graph
    vertices = parse_xml(file_path)
    graph = create_graph(vertices)
    
    # Save or plot the graph
    output_file = os.path.join(output_dir, f"{filename[:-4]}.png")
    plot_and_save_graph(graph, output_file)
    print("=====================================")


def load_optimal_costs(file_path=FILE_OPTIMAL_COSTS):
  with open(file_path, 'r') as json_file:
    loaded_optimal_costs = json.load(json_file)
  print(f"Optimal costs loaded from '{file_path}'.")
  return loaded_optimal_costs


def append_optimal_costs_to_dataframe(df, optimal_costs):
  """Append optimal costs to the main DataFrame."""
  # Create a new column for optimal costs if it doesn't exist
  if 'optimal_cost' not in df.columns:
    df['optimal_cost'] = pd.Series(dtype='float')
  
  # Loop through the DataFrame and fill the optimal costs
  for index, row in df.iterrows():
    problem_name = row['tsp_problem']
    if problem_name in optimal_costs:
      df.at[index, 'optimal_cost'] = optimal_costs[problem_name]
  
  print("Optimal costs appended to the DataFrame.")
  return df


def process_all_files_to_dataframe(data_dir=DIR_DATA):
  # Get the total number of files to process
  xml_files = [f for f in os.listdir(data_dir) if f.endswith(".xml")]
  total_files = len(xml_files)
  
  # Initialize a list to store the data
  data = []
  
  for idx, filename in enumerate(xml_files, 1):
    print(f"Processing {filename[:-4]} [{idx}/{total_files}]...")
    file_path = os.path.join(data_dir, filename)
    
    # Parse XML and create the graph
    vertices = parse_xml(file_path)
    graph = create_graph(vertices)
    num_vertices = len(vertices)
    
    # Extract the TSP problem name (assuming it can be extracted from the XML)
    tsp_problem = filename[:-4]  # Assuming the file name represents the problem
    
    # Append the data to the list
    data.append({
      'tsp_problem': tsp_problem,
      'number_of_vertices': num_vertices,
      'graph': graph
    })
    print("=====================================")
  
  # Convert the list of dictionaries to a DataFrame
  df = pd.DataFrame(data)

  # Load optimal costs from a JSON file
  loaded_optimal_costs = load_optimal_costs()

  # Append optimal costs to the DataFrame
  df = append_optimal_costs_to_dataframe(df, loaded_optimal_costs)
  
  return df



def save_dataframe_to_file(df, output_file=FILE_DATA_FRAME):
  start = time.time()
  df.to_pickle(output_file)
  end = time.time()
  print_time_taken(f"save DataFrame to {output_file}", start, end)
  print(f"DataFrame saved to {output_file}.")


def load_dataframe_from_file(input_file=FILE_DATA_FRAME):
  df = pd.read_pickle(input_file)
  print("DataFrame loaded successfully.")
  return df
