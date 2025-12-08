import os
import networkx as nx
import argparse
import gurobipy as gp
import random
import matplotlib.pyplot as plt
import csv

from collections import deque
from gurobipy import GRB
from pathlib import Path
from collections import deque

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
    return num_nodes, num_layers, G

def run_ilp(G):
    node_count = G.number_of_nodes()
    num_boundaries = num_stages - 1
    edge_count = G.number_of_edges()
    graph_edges = list(G.edges())

    epsilon = 0.05 
    avg_size = node_count / num_stages

    # Create model
    model = gp.Model('fpga_partitioning')

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

    # Print the Optimal Stage Assignment + Runtime
    print("Vertex | Optimal Stage")
    for v in node_range:
        for s in stage_range:
            # Checks if binary variable equals 1 (within a small tolerance)
            if x[v, s].X > 0.5: 
                node_label = v 
                print(f"{node_label:<6} | {s}")
                break 

    return model.objVal


def calculate_critical_path_length(G):
    cpl = {}    # critical path length
    weight = 1  # Modeling a synthetic circuit where all logic blocks are treated identical
    
    # Find primary output nodes (which have out_degree = 0)
    output_nodes = [v for v in G.nodes() if G.out_degree(v) == 0]

    # Create adjacency list of reversed graph, ensures successor nodes processed first
    reversed_adj = {v: [] for v in G.nodes()}
    for u in G.nodes():
        for v in G.successors(u):
            reversed_adj[v].append(u)
    
    # Assign critical path length of output node to weight 
    for v in output_nodes:
        cpl[v] = weight

    # Use a queue starting with the output nodes for backward traversal
    q = deque(output_nodes)

    # Count how many predecessors have had their CPL calculated
    predecessor_count = {v: 0 for v in G.nodes()}
    
    while q:
        v = q.popleft()

        # For each predecessor 'u' of node v
        for u in G.predecessors(v):
            predecessor_count[u] += 1
            
            # Checks if all successors of node u have been processed 
            if predecessor_count[u] == G.out_degree(u):
                max_successor_cpl = 0
                for successor_node in G.successors(u):  # Loop through all nodes that node u is connected to
                    max_successor_cpl = max(max_successor_cpl, cpl.get(successor_node, 0)) 
                cpl[u] = max_successor_cpl
                q.append(u)

    return cpl


def run_list_scheduling(G, p):
    stages = []
    stage_of = {}
    cur_stage = []
    cur_used = 0
    Mc = G.number_of_nodes()/p  # As specified in paper
    
    # Topologically order nodes in graph
    priority_cpl = calculate_critical_path_length(G) 
    
    # Use CPL for priority
    rank = lambda v: priority_cpl.get(v, 0)

    # Setup successor and indegree structures
    in_degree = {v: G.in_degree(v) for v in G.nodes()}
    successors = {v: list(G.successors(v)) for v in G.nodes()}

    # Create list of nodes with no incoming edges
    ready = [v for v in G.nodes() if in_degree[v] == 0]
    ready.sort(key=rank, reverse=True)

    while ready:
        v = ready.pop(0)
        w = 1

        # If it doesn't fit in current stage, start new one
        if cur_used + w > Mc:
            stages.append(cur_stage)
            cur_stage = []
            cur_used = 0

        # assign
        cur_stage.append(v)
        cur_used += w
        stage_of[v] = len(stages) + 1   # 1-based stage index

        # update successors
        for u in successors[v]:
            in_degree[u] -= 1
            if in_degree[u] == 0:
                ready.append(u)

        # maintain priority
        ready.sort(key=rank, reverse=True)

    # flush last stage
    if cur_stage:
        stages.append(cur_stage)
    
    print("Stages:", stages)
    print("Stage of:", stage_of)

    mr_count = 0
    for u, v in G.edges():
        if stage_of[u] != stage_of[v]:
            mr_count += 1
    
    return mr_count


if __name__ == "__main__":
    # # User input to directory containing graphs
    # parser = argparse.ArgumentParser(description="Process files in a directory.")
    # parser.add_argument("directory", type=str, help="The path to the directory to process.")
    # args = parser.parse_args()
    # graph_dir = args.directory
    # print(graph_dir)

    num_stages = 8 # Number of stages used in paper

    # Generate DAG
    num_nodes, num_stages, G = generate_dag()

    # Run ILP Optimization
    ilp_mr = run_ilp(G)
    print("# MRs from ILP:", ilp_mr)

    # Run List Scheduling Heuristic
    list_schedule_mr = run_list_scheduling(G, num_stages)
    print("# MRs from List Scheduling:", list_schedule_mr)

    # Save results
    current_directory = os.getcwd()
    data = [num_nodes, num_stages, ilp_mr, list_schedule_mr]
    headers = ['Nodes','Depth','ILP MRs', 'List Scheduling MRs']
    full_path = os.path.join(current_directory, 'results.csv')
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write the header row
        writer.writerows([data])
    

