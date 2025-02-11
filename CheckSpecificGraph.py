import itertools
import networkx as nx
import time
from multiprocessing import Pool, Manager
from tqdm import tqdm
from networkx.algorithms.isomorphism import GraphMatcher
from colorama import Fore, Style
from ComputeInducedRamsey import draw_graphs, Exist_coloring_avoiding





def CheckGraph(G, H, F, weak):
    """
    Takes care of visualisation, checks whether F is a valid host graph, outputs results

    Args:
        G (Graph): The graph to avoid in red.
        H (Graph): The graph to avoid in blue.
        F (Graph): The potential host graph.
        weak (bool): True, if weak induced, False if ordinary induced.
    """
    # Draw the input graphs to visually check correct inputs
    draw_graphs([G, H, F],titles=["The input graph G", "The input graph H","The potential host graph"])
    # begin timer
    start_time = time.time()
    print("\nBeginning Computations\n")
    # as we are looking for (weak) induced copies, if both graphs have edges, there must exist an induced copy of both graohs in a potential host graph.
    coloring_exist, nr_colorings = Exist_coloring_avoiding(F, G, H, weak)
    naive_colorings = 2 ** F.number_of_edges()
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
    # 2 Cases: F is host or not
    if coloring_exist:
        print(Fore.YELLOW + f"\nCannot follow anything for the {type} Ramsey number of G and H.\n")
        print(Style.RESET_ALL + f"We have found a coloring, which can avoid a red G or a blue H.")
        print(f"We have checked {nr_colorings:,} different colorings to come to this conclusion (instead of {naive_colorings:,} in a naive approach).".replace(",", "."))
        print(f"Computation time: {hours:02}:{minutes:02}:{seconds:02}.\n")
        # Visualize the result
    else:
        print(Fore.YELLOW + f"\nThe {type} Ramsey number of G and H is at most {F.number_of_nodes()}.\n")
        print(Style.RESET_ALL + f"F cannot be colored avoiding red G and blue H.\n")
        print(f"We have checked {nr_colorings:,} different colorings to come to this conclusion (instead of {naive_colorings:,} in a naive approach).".replace(",", "."))
        print(f"Computation time: {hours:02}:{minutes:02}:{seconds:02}.\n")


if __name__ == '__main__':
    # initialize the parameters
    G = nx.cycle_graph(4) # The first input graph
    H  = nx.cycle_graph(4) # The second input graph
    F = nx.complete_bipartite_graph(5,5) # The potential host graph
    weak = True # True: Weak induced Ramsey number, False: induced Ramsey number
    # start computation
    CheckGraph(G, H, F, weak)