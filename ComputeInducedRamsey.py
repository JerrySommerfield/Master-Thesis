import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, Manager
from tqdm import tqdm
from networkx.algorithms.isomorphism import GraphMatcher
from colorama import Fore, Style


def draw_graphs(graphs, edge_colors=None, node_color="lightgray", node_size=300, with_labels=True, positions=None, edge_width=1, titles=None):
    """
    Draw up to three graphs, optionally side by side.

    Args:
        graphs (list of nx.Graph): List of graphs to draw.
        edge_colors (list of dict): List of edge color dictionaries, one for each graph (default: None).
        node_color (str): Color of nodes for all graphs (default: "lightgray").
        node_size (int): Size of the nodes (default: 300).
        with_labels (bool): Whether to show labels on the nodes (default: True).
        positions (list of dict): Custom positions for nodes in each graph (default: None).
        edge_width (float): Width of the edges (default: 1).
        titles (list of str): Titles for each graph (default: None).

    Returns:
        list of dict: Positions used for the graphs.
    """
    # Validate inputs
    num_graphs = len(graphs)
    if num_graphs > 3:
        raise ValueError("Can only draw up to 3 graphs.")
    if edge_colors is None:
        edge_colors = [None] * num_graphs
    if positions is None:
        positions = [None] * num_graphs
    if titles is None:
        titles = [None] * num_graphs
    # Generate default layouts for graphs if not provided
    positions = [
        nx.spring_layout(graph) if pos is None else pos
        for graph, pos in zip(graphs, positions)
    ]
    # Prepare figure
    fig, axes = plt.subplots(1, num_graphs, figsize=(8 * num_graphs, 8))
    if not isinstance(axes, (list, np.ndarray)):  # Ensure axes is iterable
        axes = [axes]
    # Helper function to draw a single graph
    def draw_single_graph(ax, graph, edge_colors, pos, title):
        ax.axis("off")
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_color, node_size=node_size)
        if edge_colors:
            for color, edges in edge_colors.items():
                nx.draw_networkx_edges(
                    graph, pos, ax=ax, edgelist=edges, edge_color=color, width=edge_width
                )
        else:
            nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_width)
        if with_labels:
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10, font_color="black")
        if title:
            ax.set_title(title, fontsize=16)
    # Draw each graph
    for ax, graph, edges, pos, title in zip(axes, graphs, edge_colors, positions, titles):
        draw_single_graph(ax, graph, edges, pos, title)
    # Set a custom title for the window
    plt.show()
    return positions


def Abort_partial_coloring(G, H, F, Red, Blue, Uncolored_red, Uncolored_blue, weak):
    """
    Checks for red G or blue H in given partial coloring, which cannot be broken later. (Induced or weak induced)

    Args:
        G (Graph): The graph to check for in red.
        H (Graph): The graph to check for in blue.
        F (Graph): The potential host graph.
        Red (Graph): The partial red subgraph.
        Blue (Graph): The partial blue subgraph.
        Uncolored_red (Graph): Red + uncolored subgraph.
        Uncolored_blue (Graph): Blue + uncolored subgraph.
        weak (bool): True if weak induced, False if ordinary induced.

    Returns:
        bool: True if there exists such a copy of G in red or of H in blue, in this case coloring can be aborted, else False.
    """
    # Use GraphMatcher to find all subgraphs of the red graph isomorphic to G
    for mapping in GraphMatcher(Red, G).subgraph_isomorphisms_iter():
        # Check if induced subgraph on same vertices in F (ind) or if uncolored edges are added is still isomorphic to G
        # This ensures it cannot later be "broken" by adding more red edges
        if weak: # search in the uncolored + red graph
            if nx.is_isomorphic(Uncolored_red.subgraph(mapping.keys()), G):
                return True  # If such a subgraph exists, return True immediately
        else: # search in F
            if nx.is_isomorphic(F.subgraph(mapping.keys()), G):
                return True  # If such a subgraph exists, return True immediately  
    # Use GraphMatcher to find all subgraphs of the blue graph isomorphic to H
    for mapping in GraphMatcher(Blue, H).subgraph_isomorphisms_iter():
        # Check if induced subgraph on same vertices in F (ind) or if uncolored edges are added is still isomorphic to H
        # This ensures it cannot later be "broken" by adding more blue edges
        if weak: # search in the uncolored + blue graph
            if nx.is_isomorphic(Uncolored_blue.subgraph(mapping.keys()), H):
                return True  # If such a subgraph exists, return True immediately
        else: # search in F
            if nx.is_isomorphic(F.subgraph(mapping.keys()), H):
                return True  # If such a subgraph exists, return True immediately
    # If no such subgraph is found in either color, return False
    return False


def Recursive_coloring(G, H, F, Red, Blue, Uncolored, Uncolored_red, Uncolored_blue, weak, number=0):
    """
    Recursively generates all colorings of partially colored graph avoiding red G and blue H. (Induced or weak induced)

    Args:
        G (Graph): The graph to check for in red.
        H (Graph): The graph to check for in blue.
        F (Graph): The potential host graph.
        Red (Graph): The red subgraph.
        Blue (Graph): The blue subgraph.
        Uncolored (Graph): The uncolored subgraph.
        Uncolored_red (Graph): Red + uncolored subgraph.
        Uncolored_blue (Graph): Blue + uncolored subgraph.
        weak (bool): True if weak induced, Flase if ordinary induced.
        number (int): Counter for the number of iterations.

    Returns:
        bool:   True if valid coloring is found without red G or blue H (with partial coloring intact).
                False if red G or blue H cannot be avoided (with partial coloring intact).
        int: The number of (partial) colorings checked.
    """
    number += 1
    # Check if we can abort current coloring (red G or blue H found)
    if Abort_partial_coloring(G, H, F, Red, Blue, Uncolored_red, Uncolored_blue, weak):
        return False, number
    # If no edges remain uncolored, return success (no red G or blue H)
    if Uncolored.number_of_edges() == 0:
        return True, number
    # Get the next edge to color
    next_edge = list(Uncolored.edges())[0]
    # Create copies of the graphs: Changes should only be made to copies!
    Red_plus, Blue_plus = Red.copy(), Blue.copy()
    Uncolored_minus, Uncolored_red_minus, Uncolored_blue_minus = Uncolored.copy(), Uncolored_red.copy(), Uncolored_blue.copy()
    # Remove the edge from all relevant uncolored copies
    Uncolored_minus.remove_edge(*next_edge)
    Uncolored_red_minus.remove_edge(*next_edge)
    Uncolored_blue_minus.remove_edge(*next_edge)
    # Update Red_copy and Blue_copy
    Red_plus.add_edge(*next_edge)
    Blue_plus.add_edge(*next_edge)
    # Recursive call for Case 1: Edge was added to red graph and removed from uncolored and uncolored_blue
    coloring_exists, number = Recursive_coloring(
        G, H, F, Red_plus, Blue, Uncolored_minus, Uncolored_red, Uncolored_blue_minus, weak, number
    )
    # return results if coloring exists. Otherwise continue with the other case
    if coloring_exists:
        return coloring_exists, number
    # Recursive call for Case 2: Edge was added to blue graph and removed from uncolored and uncolored_red
    coloring_exists, number = Recursive_coloring(
        G, H, F, Red, Blue_plus, Uncolored_minus, Uncolored_red_minus, Uncolored_blue, weak, number
    )
    # return results
    return coloring_exists, number


def Exist_coloring_avoiding(F, G, H, weak):
    """
    Runs the coloring process to determine if red G or blue H can be avoided in F. (Induced or weak induced)

    Args:
        F (Graph): The graph to be colored (not empty).
        G (Graph): The graph to check for in red (not empty).
        H (Graph): The graph to check for in blue (not empty).
        weak (bool): True if weak induced, False if ordinary induced.

    Returns:
        bool: True if a valid coloring avoiding both graphs is found, False otherwise.
        int: The number of (partial) colorings checked.
    """
    # Initialize graphs
    Red = nx.Graph()  # The red subgraph starts empty
    Red.add_nodes_from(F.nodes())  # It has the same nodes as F
    Blue = Red.copy()  # The blue subgraph also starts empty
    Uncolored = F.copy()  # Initially, all edges are uncolored
    Uncolored_red = F.copy()  # Red + uncolored graph
    Uncolored_blue = F.copy()  # Blue + uncolored graph
    # Start coloring with the first uncolored edge. If G=H, it can w.l.o.g. be colored red, as it needs some color (Avoids checking each coloring twice with swapped colors)
    if G == H:
        first_edge = list(Uncolored.edges())[0]
        Red.add_edge(*first_edge)  # Color this edge red
        Uncolored.remove_edge(*first_edge)  # Remove the edge from uncolored edges
        Uncolored_blue.remove_edge(*first_edge)  # Update blue + uncolored graph
    # Perform the recursive coloring process and return the results
    return Recursive_coloring(
        G, H, F, Red, Blue, Uncolored, Uncolored_red, Uncolored_blue, weak
    )


def check_graph_mp(args):
    """
    Multiprocessing-compatible function to check whether a graph is a host graph for G and H. 
    Creates F and uses Exist_color_avoiding() to color it. Returns whether a coloring of F exists, avoiding red G and blue H.

    Args:
        args (tuple): Contains 
            edges (list): List of the edges for host graph.
            G (Graph): Graph to avoid in red.
            H (Graph): Graph to avoid in blue.
            m (int): Number of vertices of the host graph.
            weak(bool): True if weak induced, False if ordinary induced.
            check_H_induced (bool): True if potential host graphs should be checked for induced H, False otherwise.
            stop_flag: shared variable to tell the pool to terminate.

    Returns:
        tuple: 
            bool: True if the graph on 'edges' is a host graph, False otherwise.
            Graph: The graph on 'edges' if it is a host graph, None otherwise.
            int: The number of colorings checked.
            int: The number of colorings one would have naively checked.
    """
    edges, G, H, m, weak, check_H_induced, stop_flag = args
    # Check the stop flag before starting computation
    if stop_flag.get("found", False):
        return False, None, 0, 0 # Return imediatly
    # Generate the potential host graph
    F = nx.empty_graph(m)
    F.add_edges_from(edges)  # Add the edges
    
    if check_H_induced:
        # Check, whether there exists an induced copy of H (otherwise color all edges blue) (We already ensured that G exists induced in F)
        if not GraphMatcher(F, H).subgraph_is_isomorphic():
            return False, None, 0, 2 ** len(edges)
    
    # Check if the graph is a host graph by checking if it can be colored avoiding red G and blue H.
    found_coloring, colorings_checked = Exist_coloring_avoiding(F, G, H, weak)
    if not found_coloring:  # If F is a host graph (coloring avoiding not found)
        stop_flag["found"] = True  # Update the shared flag
        return True, F, colorings_checked, 2 ** len(edges) # return F
    return False, None, colorings_checked, 2 ** len(edges) # return that F is not a host graph


def Find_Host_Graph_on_m_mp(G, H, m, edges_to_include, edges_to_exclude, weak, check_H_induced):
    """
    Multiprocessing function that itterates over all edgesets of potential host graphs and uses check_graph_mp() to check them with early termination when a solution is found.

    Args:
        G (Graph): The graph to avoid in red.
        H (Graph): The graph to avoid in blue.
        m (int): Number of vertices in the generated graphs.
        edges_to_include (list): Edges that must always be included in the graph.
        edges_to_exclude (list): Edges that must always be excluded from the graph.
        weak (bool): True if weak induced, False if ordinary induced.
        check_H_induced (bool): True if potential host graphs should be checked for induced H, False otherwise.

    Returns:
        bool: True if a host graph is found, False otherwise.
        Graph: The host graph if it exists, None otherwise.
        int: The number of colorings checked.
        int: The number of colorings we would have naively checked.
    """
    # Generate a list of optional edges
    complete_graph = nx.complete_graph(m)
    optional_edges = set(complete_graph.edges()) - set(edges_to_include) - set(edges_to_exclude)
    # Generate all subsets of optional edges            ---> This may be unwise, as all combinations must be saved. May be better to generate as we go and delete when no longer needed
    edge_combinations = list(itertools.chain.from_iterable(
        itertools.combinations(optional_edges, r) for r in range(len(optional_edges) + 1)
    ))
    # Shared dictionary for early stopping
    manager = Manager()
    stop_flag = manager.dict({"found": False})  # Shared state between processes - if "found" == True, the other processes can be stopped
    # Prepare arguments for the multiprocessing pool   ---> Again not optimal for Memory
    tasks = [
        (list(edges) + list(edges_to_include), G, H, m, weak, check_H_induced, stop_flag)
        for edges in edge_combinations
    ]
    # initialize variables
    colorings_total = 0
    naive_total = 0
    # Parallel processing with tqdm
    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(check_graph_mp, tasks), total=len(tasks), desc=f"Checking all graphs on {m} vertices"):
            is_host, host_graph, colorings_checked, naive_checked = result
            # update Variables
            colorings_total += colorings_checked
            naive_total += naive_checked
            if is_host:
                pool.terminate()  # Stop the pool as we found the result
                return True, host_graph, colorings_total, naive_total
    return False, None, colorings_total, naive_total


def Special_cases(G, H):
    """
    Checks some trivial cases (empty graphs)

    Args:
        G (Graph): The graph to avoid in red.
        H (Graph): The graph to avoid in blue.

    Returns:
        bool: True if a host graph was found, otherwise False.
        Graph: Host graph if at least one graph was empty, otherwise None.
    """
    if G.size() == 0 and H.size() == 0: # host is smaller graph 
        if G.number_of_nodes() <= H.number_of_nodes():
            return True, G
        return True, H
    elif G.size() == 0: # host is G
        return True, G
    elif H.size() == 0: # host is H
        return True, H
    else: # host is not obvious
        return False, None


def Compute_induced_Ramsey(G_in, H_in, n, weak):
    """
    Takes care of visualisation, computes the ramsey number if possible, outputs the results

    Args:
        G (Graph): The graph to avoid in red.
        H (Graph): The graph to avoid in blue.
        n (int): The number of vertices for the host graph.
        weak (bool): True, if weak induced, False if ordinary induced.
    """
    # We want G to be at least as large as H, so we may change the order (This speeds up computation in a couple of spots)
    if G_in.number_of_nodes() >= H_in.number_of_nodes():
        G, H = G_in, H_in
    else:
        G, H = H_in, G_in
    # Draw the input graphs to visually check correct inputs
    draw_graphs([G, H],titles=["The input graph G", "The input graph H"])
    # begin timer
    start_time = time.time()
    print("\nBeginning Computations\n")
    # as we are looking for (weak) induced copies, if both graphs have edges, there must exist an induced copy of both graohs in a potential host graph.
    found_Host, Host_graph = Special_cases(G, H) #checks for empty graphs.
    colorings_total, naive_total = 0, 0
    if not found_Host: # None of the graphs are empty. Thus in a potential Host grpah, both graphs must exist as induced subgraphs (otherwise color everything in the other color.)
        # define the edges we want to always include or always exclude (there must be an induced copy of G in the host graph, otherwise color it all red)
        include = set(G.edges())
        exclude = set(nx.complete_graph(G.nodes()).edges()) - set(G.edges()) # these are the non-edges of G
        m = G.number_of_nodes()
        # check if H is not induced in G
        check_H_induced = not(GraphMatcher(G, H).subgraph_is_isomorphic())
        # now begin searching every coloring of every suitable graph in order of vertices
        while not(found_Host) and m <= n:
            found_Host, Host_graph, colorings_on_m, naive_on_m = Find_Host_Graph_on_m_mp(G, H, m, include, exclude, weak, check_H_induced)
            colorings_total += colorings_on_m
            naive_total += naive_on_m
            m += 1
    # stop the timer
    end_time = time.time()
    duration = end_time - start_time
    # Convert elapsed time to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    # for the output text:
    if weak:
        type = "weak induced"
    else:
        type = "induced"
    # Output the results
    # 2 Cases: Host was found or not
    if found_Host: 
        # If the host graph is empty, at least one of the inputs must have been empty.
        # The means by which we computed the (weak) induced Ramsey number was different, which warrents a different output
        if Host_graph.number_of_edges() == 0: 
            print(Fore.YELLOW + f"\nThe {type} Ramsey number of G and H is {Host_graph.number_of_nodes()}.\n")
            print(Style.RESET_ALL + f"This was a trivial case involving an empty input graph.\n")
        else:
            print(Fore.YELLOW + f"\nThe {type} Ramsey number of G and H is {Host_graph.number_of_nodes()}.\n")
            print(Style.RESET_ALL + f"We have found a graph on {Host_graph.number_of_nodes()} vertices, which no matter the coloring cannot avoid a red G or a blue H.")
            print(f"All suitable graphs on up to {Host_graph.number_of_nodes()-1} vertices were also tested and there was none, that did not have a coloring avoiding red G and blue H.\n")
            print(f"We have checked {colorings_total:,} different colorings to find the graph (instead of {naive_total:,} in a naive approach).\n".replace(",", "."))
            print(f"Computation time: {hours:02}:{minutes:02}:{seconds:02}.\n")
        # Visualize the result
        draw_graphs([G,H,Host_graph],titles=["The input graph G","The input graph H",f"Host graph ({type})"])
    else:
        print(Fore.YELLOW + f"\nThe {type} Ramsey number of G and H is at least {n+1}.\n")
        print(Style.RESET_ALL + f"All suitable graphs on up to {n} vertices were tested and there was none, that did not have a coloring avoiding red G and blue H.\n")
        print(f"We have checked {colorings_total:,} different colorings to come to this conclusion (instead of {naive_total:,} in a naive approach).".replace(",", "."))
        print(f"Computation time: {hours:02}:{minutes:02}:{seconds:02}.\n")


if __name__ == '__main__':
    # initialize the parameters
    n = 8 # The number of vertices up to which we should check for host graphs
    G_in = nx.complete_graph(4) # The first input graph
    H_in = nx.complete_graph(3)
    H_in.add_edges_from([(0,3)]) # The second input graph
    weak = True # True: Weak induced Ramsey number, False: induced Ramsey number
    # start computation
    Compute_induced_Ramsey(G_in, H_in, n, weak)