import networkx as nx

if True:

    # edge + vertex
    G = nx.complete_graph(2)
    G.add_nodes_from([2])

    # 3 path
    G = nx.path_graph(3)

    # 3 complete
    G = nx.complete_graph(3)

    # edge + 2 vertices
    G = nx.complete_graph(2)
    G.add_nodes_from([2,3])

    # matching
    G = nx.complete_graph(2)
    G.add_edges_from([(2,3)])

    # L
    G = nx.path_graph(3)
    G.add_nodes_from([3])

    # 4 path
    G = nx.path_graph(4)

    # Star
    G = nx.star_graph(3)

    # triangle + vertex
    G = nx.complete_graph(3)
    G.add_nodes_from([3])

    # 4 cycle
    G = nx.cycle_graph(4)

    # triangle plus edge
    G = nx.complete_graph(3)
    G.add_edges_from([(0,3)])

    # complete - edge
    G = nx.cycle_graph(4)
    G.add_edges_from([(0,2)])

    # complete
    G = nx.complete_graph(4)