import math 
import random
import networkx as nx

def Interface_Graph(sizes, p_in, p_out, number_node_in_contact, seed=None):
    """Return the random partition graph with a partition of sizes.

    A partition graph is a graph of communities with sizes defined by
    s in sizes. Nodes in the same group are connected with probability
    p_in and nodes of different groups are connected with probability
    p_out.
    
    It is a modification of the code found in networkx. Here only 
    number_node_in_contact in each block from each block are in 
    contact with node of other block.
    
    Parameters
    ----------
    sizes : list of ints
      Sizes of groups
    p_in : float
      probability of edges with in groups
    p_out : float
      probability of edges between groups
    number_node_in_contact : int
        number of node of each block which are going to form link with other group
    directed : boolean optional, default=False
      Whether to create a directed graph
    seed : int optional, default None
      A seed for the random number generator

    Returns
    -------
    G : NetworkX Graph or DiGraph
      random partition graph of size sum(gs)

    Raises
    ------
    NetworkXError
      If p_in or p_out is not in [0,1]

       """
    # Use geometric method for O(n+m) complexity algorithm
    # partition=nx.community_sets(nx.get_node_attributes(G,'affiliation'))
    directed=False
    if not seed is None:
        random.seed(seed)
    if not 0.0 <= p_in <= 1.0:
        raise nx.NetworkXError("p_in must be in [0,1]")
    if not 0.0 <= p_out <= 1.0:
        raise nx.NetworkXError("p_out must be in [0,1]")

    G = nx.Graph()
    G.name = "Interface"
    G.graph['partition'] = []
    n = sum(sizes)
    G.add_nodes_from(range(n))
    # start with len(sizes) groups of gnp random graphs with parameter p_in
    # graphs are unioned together with node labels starting at
    # 0, sizes[0], sizes[0]+sizes[1], ...
    next_group_false = {}  # maps node key (int) to first node in next group
    start = 0
    start_false = 0
    group = 0
    for n in sizes:
        edges = ((u+start, v+start)
                 for u, v in
                 nx.fast_gnp_random_graph(n, p_in, directed=directed).edges())
        G.add_edges_from(edges)
        next_group_false.update(dict.fromkeys(range(start_false, start_false+number_node_in_contact), start_false+number_node_in_contact))
        G.graph['partition'].append(set(range(start, start+n)))
        group += 1
        start_false += number_node_in_contact
        start += n
    L_sizes = [0]
    for i in range(len(sizes)-1):
        L_sizes.append(L_sizes[i]+sizes[i])
    ###
    # handle edge cases
    if p_out == 0:
        return G
    if p_out == 1:
        print(next_group_false)
        for n in next_group_false:
            targets = range(next_group_false[n], len(sizes)*number_node_in_contact)
            n = n//number_node_in_contact*L_sizes[n//number_node_in_contact] + n%number_node_in_contact
            New_targets = []
            for i in targets:
                New_targets.append( i//number_node_in_contact*L_sizes[i//number_node_in_contact] + i%number_node_in_contact ) 
            G.add_edges_from(zip([n]*len(New_targets), New_targets))
        return G
    # connect each node in group randomly with the nodes not in group
    # use geometric method like fast_gnp_random_graph()
    ### modif

    lp = math.log(1.0 - p_out)
    n = number_node_in_contact*len(sizes)#len(G)
    for u in range(n-1):
        v = next_group_false[u]  # start with next node not in this group
        while v < n:
            lr = math.log(1.0 - random.random())
            v += int(lr/lp)
            if v < n:
                G.add_edge(u//number_node_in_contact*L_sizes[u//number_node_in_contact] + u%number_node_in_contact, v//number_node_in_contact*L_sizes[v//number_node_in_contact] + v%number_node_in_contact)
                v += 1
    return G