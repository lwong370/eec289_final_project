import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_dag( num_nodes=12, num_layers=4, edge_prob=0.25,seed=None):

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
        G, layers = generate_dag()


    assert nx.is_directed_acyclic_graph(G)
    return G, layers


#if __name__ == "__main__":
#    G, layers = generate_dag()
#    print("\nGenerated  DAG:")
#    print("Layers:", layers)
#    print("Nodes:", list(G.nodes()))
#    print("Edges:")
#    for u, v in G.edges():
#        print(f"  {u} -> {v}")
#
#    plt.figure(figsize=(6,5))
#    pos = nx.spring_layout(G, seed=1)
#    nx.draw(G)
#    plt.title("FPGA-like DAG (2-terminal nets)")
#    plt.show()



