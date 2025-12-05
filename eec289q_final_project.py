import os
import networkx as nx
import argparse
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path


if __name__ == "__main__":

    # Create model
    model = gp.Model('fpga_partitioning')

    # User input to directory containing graphs
    parser = argparse.ArgumentParser(description="Process files in a directory.")
    parser.add_argument("directory", type=str, help="The path to the directory to process.")
    args = parser.parse_args()
    graph_dir = args.directory
    print(graph_dir)

    num_stages = 8 # Number of stages used in paper

    # Iterate over all graph files in directory
    for graph_file in Path(graph_dir).glob('*.graphml'):
        # Load graph from file name
        #print(graph_file.name)
        Graph = nx.read_graphml(graph_file)
        node_count = Graph.number_of_nodes()
        num_boundaries = num_stages - 1
        edge_count = Graph.number_of_edges()
        graph_edges = list(Graph.edges())
        #print(type(Graph), Graph.number_of_nodes(), Graph.number_of_edges())

        # Ensure they are directed and acyclic
        if not Graph.is_directed():
            Graph = Graph.to_directed()

        # Skip non-DAGs
        if not nx.is_directed_acyclic_graph(Graph):
            #print("not a DAG, skipping")
            continue

        # Variable definition: Creates a 2-dimensional variable x[v, s] = 1 if vertex v in stage s
        node_range = range(1, node_count+1)
        stage_range = range(num_stages)
        x = model.addVars(node_range, stage_range, vtype=GRB.BINARY, name="x")

        # Variable definition: micro-register mr[u,v,k] at a stage boundary k between vertex u and v
        mr = model.addVars(graph_edges, num_boundaries, vtype=GRB.BINARY, name="mr")

        # Constraint: Ensures each vertex is in only 1 stage
        for v in range(node_count):
            model.addConstr(gp.quicksum(x[v,s] for s in range(num_stages)) == 1)

        # Constraint: Links stage to an MR
        for (u,v) in graph_edges:
            for k in range(node_range - 1):
                for s in range(k+1,stage_range): 
                    model.addConstr(mr[u,v, k] >= x[v,s])

                for s in range(0,k+1):
                    model.addConstr(mr[u,v, k] >= x[v,s])
        
        # Objective function: Minimize total MR's needed 
        model.setObjective(gp.quicksum(mr[u,v,k] for (u,v) in graph_edges for k in range(num_boundaries)), GRB.MINIMIZE)
        model.optimize()
        print(f"Runtime: {model.Runtime}")