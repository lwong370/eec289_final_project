import os
import networkx as nx
import argparse
import gurobipy as gp
import random
import matplotlib.pyplot as plt

from gurobipy import GRB
from pathlib import Path


def generate_dag( num_nodes=100, num_layers=10, edge_prob=0.25,seed=None):
    base_prob=0.05
    decay=0.3
    if seed is not None:
        random.seed(seed)

    # make an empty directed graph, with num_nodes number of nodes
    G = nx.DiGraph()
    nodes = list(range(num_nodes))
    G.add_nodes_from(nodes)

    # assign nodes to layers
    layers = [[] for _ in range(num_layers)]
    for layer_idx, node in enumerate(nodes[:num_layers]): #at least one node per layer
        layers[layer_idx].append(node)
    for node in nodes[num_layers:]: # assign rest of layers randomly
        L = random.randint(0, num_layers - 1)
        layers[L].append(node)

    # edges: probability decays with layer distance
    for Li in range(num_layers):
        for Lj in range(Li + 1, num_layers):
            dist = Lj - Li #calc distance between layer and other layers
            p = base_prob * (decay ** (dist - 1))  # decay the base probability depending on how far it is
            for u in layers[Li]: #look at all node pairs
                for v in layers[Lj]:
                    if random.random() < p:
                        G.add_edge(u, v)

    ## Fix up connectivity so nodes aren't useless ##

    # make sure every non-input node has at least one parent
    for L in range(1, num_layers):
        possible_parents = []
        for Li in range(0, L):
            possible_parents.extend(layers[Li])

        for v in layers[L]:
            if G.in_degree(v) == 0 and possible_parents:
                u = random.choice(possible_parents)
                G.add_edge(u, v)

    # make sure every non-output node has at least one child
    for L in range(0, num_layers - 1):
        possible_children = []
        for Lj in range(L + 1, num_layers):
            possible_children.extend(layers[Lj])

        for u in layers[L]:
            if G.out_degree(u) == 0 and possible_children:
                v = random.choice(possible_children)
                G.add_edge(u, v)
    
    # make sure the DAG is weakly connected
    if not nx.is_weakly_connected(G):
        # regenerate until connected
        G = generate_dag()

    assert nx.is_directed_acyclic_graph(G)
    return G


if __name__ == "__main__":

    # Create model
    model = gp.Model('fpga_partitioning')

    # # User input to directory containing graphs
    # parser = argparse.ArgumentParser(description="Process files in a directory.")
    # parser.add_argument("directory", type=str, help="The path to the directory to process.")
    # args = parser.parse_args()
    # graph_dir = args.directory
    # print(graph_dir)

    num_stages = 8 # Number of stages used in paper

    G = generate_dag()

    node_count = G.number_of_nodes()
    num_boundaries = num_stages - 1
    edge_count = G.number_of_edges()
    graph_edges = list(G.edges())

    epsilon = 0.05 
    avg_size = node_count / num_stages

    # Variable definition: Creates a 2-dimensional variable x[v, s] = 1 if vertex v in stage s
    node_range = range(node_count)
    stage_range = range(num_stages)
    x = model.addVars(node_range, stage_range, vtype=GRB.BINARY, name="x")

    # Variable definition: micro-register mr[u,v,k] = 1 if the edge between vertex u and v cross stage boundary k
    mr = model.addVars(graph_edges, num_boundaries, vtype=GRB.BINARY, name="mr")
    
    # Variable definition: max capacity per stage
    mc = model.addVar(vtype=GRB.INTEGER, name="mc")

    # Constraint: Ensures each vertex is in only 1 stage
    for v in node_range:
        model.addConstr(gp.quicksum(x[v,s] for s in range(num_stages)) == 1)

    # Constraint: Makes sure that we limit the number of nodes that are assigned to any one stage
    model.addConstr(mc >= (1 - epsilon) * avg_size) # Calculation fir mc specified in paper
    model.addConstr(mc <= (1 + epsilon) * avg_size)
    for s in range(num_stages):
        model.addConstr(gp.quicksum(x[v, s] for v in node_range) <= mc)

    # Constraint: Forces mr to be 1 if edge (u,v) crosses boundary k  
    for (u,v) in graph_edges:
        for k in range(num_boundaries - 1):
            sum_u_before = gp.quicksum(x[u, s] for s in range(k + 1)) 
            sum_v_after = gp.quicksum(x[v, s] for s in range(k + 1, num_stages)) 
            
            # Forces mr[u,v,k] = 1 if the edge crosses forward
            model.addConstr(mr[u, v, k] >= sum_u_before + sum_v_after - 1)

    # Constraint: Presidence constraint -- Stage(u) <= Stage(v) 
        # aka stage of node u comes before stage of node v
    for (u, v) in graph_edges:
        stage_u = gp.quicksum(stage_range[s] * x[u, s] for s in range(num_stages)) # Gets stage where node u is placed       
        stage_v = gp.quicksum(stage_range[s] * x[v, s] for s in range(num_stages)) # Gets stage where node v is placed

        model.addConstr(stage_u <= stage_v)
    
    # Objective function: Minimize total MR's needed 
    model.setObjective(gp.quicksum(mr[u,v,k] for (u,v) in graph_edges for k in range(num_boundaries)), GRB.MINIMIZE)
    model.Params.TimeLimit = 120 # Set time limit to 2 minutes
    model.optimize()

    # Print the Optimal Stage Assignment (Variable X) ---
    print("Vertex | Optimal Stage")

    # Prints variables of x[v, s] that equal 1, meaning vertex v is in stage s
    for v in node_range:
        for s in stage_range:
            # Checks if binary variable equals 1 (within a small tolerance)
            if x[v, s].X > 0.5: 
                node_label = v 
                print(f"{node_label:<6} | {s}")
                break 

    print(f"Runtime: {model.Runtime}")
    print(f"Total Minimum Micro-Registers (MRs): {model.objVal}")

