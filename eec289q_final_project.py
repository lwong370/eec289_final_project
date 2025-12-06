import os
import networkx as nx
import argparse
import gurobipy as gp
import random
import matplotlib.pyplot as plt

from gurobipy import GRB
from pathlib import Path


def generate_dag( num_nodes=1000, num_layers=20, edge_prob=0.25,seed=None):
    base_prob=0.3 
    decay=0.5
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

    # Variable definition: Creates a 2-dimensional variable x[v, s] = 1 if vertex v in stage s
    node_range = range(node_count)
    stage_range = range(num_stages)
    x = model.addVars(node_range, stage_range, vtype=GRB.BINARY, name="x")

    # Variable definition: micro-register mr[u,v,k] = 1 if the edge between vertex u and v cross stage boundary k
    mr = model.addVars(graph_edges, num_boundaries, vtype=GRB.BINARY, name="mr")

    # Constraint: Ensures each vertex is in only 1 stage
    for v in node_range:
        model.addConstr(gp.quicksum(x[v,s] for s in range(num_stages)) == 1)

    # Constraint: Links stage to an MR
    for (u,v) in graph_edges:
        for k in range(num_boundaries - 1):
            # Old code in loop
            # for s in range(k+1,num_boundaries): 
            #     model.addConstr(mr[u,v, k] >= x[v,s])

            # for s in range(0,k+1):
            #     model.addConstr(mr[u,v, k] >= x[v,s])

            # New code in loop
            sum_u_before = gp.quicksum(x[u, s] for s in range(k + 1)) 

            # Sink (v) is AFTER the boundary k (stage k+1 to num_stages-1)
            sum_v_after = gp.quicksum(x[v, s] for s in range(k + 1, num_stages)) 
            
            # Add a constraint forcing mr[u,v,k] = 1 if the edge crosses forward
            model.addConstr(mr[u, v, k] >= sum_u_before + sum_v_after - 1)
    
    # Objective function: Minimize total MR's needed 
    model.setObjective(gp.quicksum(mr[u,v,k] for (u,v) in graph_edges for k in range(num_boundaries)), GRB.MINIMIZE)
    model.optimize()
    print(f"Runtime: {model.Runtime}")

    # Print results
    print("\n## 1. Optimal Objective (Minimum MRs) ##")
    print(f"Total Minimum Micro-Registers (MRs): {model.objVal}")

    # --- 2. Print the Optimal Stage Assignment (Variable X) ---
    print("\n## 2. Vertex-to-Stage Assignment (x[v, s]) ##")
    print("Vertex | Optimal Stage")
    print("-------|--------------")

    # The variable x[v, s] is 1 if vertex v is in stage s.
    # We only want to print the stage where x[v, s] == 1.
    for v in node_range:
        # Iterate through all possible stages for the current vertex v
        for s in stage_range:
            # Check if the binary variable is set to 1 (within a small tolerance)
            if x[v, s].X > 0.5: 
                # Note: G.nodes[v] may contain node names if the graph was built that way, 
                # otherwise, 'v' is the simple node index.
                try:
                    node_label = G.nodes[v]['label'] # Use node label if available
                except:
                    node_label = v # Otherwise, use the index
                    
                print(f"{node_label:<6} | {s}")
                break # Move to the next vertex once its stage is found

    # --- 3. Print the Required Micro-Registers (Variable MR) ---
    print("\n## 3. Required Micro-Registers (mr[u, v, k]) ##")
    print("Edge (u, v) | Stage Boundary k | MR Required (1/0)")
    print("------------|------------------|------------------")

    # The variable mr[u, v, k] is 1 if an MR is required for edge (u, v) at boundary k.
    for (u, v) in graph_edges:
        for k in range(num_boundaries):
            if mr[u, v, k].X > 0.5:
                print(f"({u}, {v}):<10 | {k:<16} | 1")
