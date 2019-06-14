import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import write_dot, graphviz_layout

from var_system import (random_gnp_dag, random_tree_dag)

def draw_graph(G, file_name=None):
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, arrowsize=30,
            with_labels=True, linewidth=2,
            font_size=24, font_family="serif",
            font_weight="bold", linewidths=2,
            alpha=0.9, node_color="lightsalmon")

    if file_name is not None:
        if isinstance(file_name, str):
            plt.savefig(file_name)
        else:  # Allow passing a list
            [plt.savefig(f) for f in file_name]
    plt.show()
    return


def draw_graph_estimates(G, G_hat, file_name=None,
                         count_self_loops=False):
    # To ensure we don't mutate them
    _G = G.copy()
    _G_hat = G_hat.copy()

    nodes = _G.nodes
    if not count_self_loops:
        _G.remove_edges_from([(i, i) for i in nodes if (i, i) in G.edges])
        _G_hat.remove_edges_from([(i, i) for i in G_hat.nodes
                                  if (i, i) in G_hat.edges])

    pos = graphviz_layout(_G, prog="dot")

    G_correct = nx.DiGraph()
    G_correct.add_nodes_from(nodes)
    G_correct.add_edges_from(set(_G.edges) & set(_G_hat.edges))

    G_missed = nx.DiGraph()
    G_missed.add_nodes_from(nodes)
    G_missed.add_edges_from(set(_G.edges) - set(_G_hat.edges))

    G_extra = nx.DiGraph()
    G_extra.add_nodes_from(nodes)
    G_extra.add_edges_from(set(_G_hat.edges) - set(_G.edges))

    n_edges = len(_G.edges)
    n_correct = len(G_correct.edges)
    n_missed = len(G_missed.edges)
    n_extra = len(G_extra.edges)
    assert n_missed == n_edges - n_correct
    

    nx.draw(G_correct, pos, arrowsize=20,
            with_labels=True, linewidth=2,
            font_size=24, font_family="serif",
            font_weight="bold",
            alpha=0.9, node_color="lightblue",
            edge_color="g")

    nx.draw(G_missed, pos, arrowsize=20,
            with_labels=True, linewidth=2,
            font_size=24, font_family="serif",
            font_weight="bold",
            alpha=0.9, node_color="lightblue",
            edge_color="r")

    nx.draw(G_extra, pos, arrowsize=20,
            with_labels=True, linewidth=2,
            font_size=24, font_family="serif",
            font_weight="bold",
            alpha=0.9, node_color="lightblue",
            edge_color="m")

    plt.plot([], [], color="g",
             label="Correctly Recovered ({} / {})"
             "".format(n_correct, n_edges))
    plt.plot([], [], color="r",
             label="False Negatives ({} / {})"
             "".format(n_missed, n_edges))
    plt.plot([], [], color="m",
             label="False Positives ({})"
             "".format(n_extra))
    plt.legend(loc="upper left")

    if file_name is not None:
        if isinstance(file_name, str):
            plt.savefig(file_name)
        else:  # Allow passing a list
            [plt.savefig(f) for f in file_name]
    plt.show()
    return


def draw_example_graphs():
    n_nodes = 75
    edge_prob = 2. / n_nodes

    G_tree = random_tree_dag(n_nodes, p_lags, pole_rad=0.75)
    G_dag = random_gnp_dag(n_nodes, p_lags, pole_rad=0.75,
                           edge_prob=edge_prob)
    G_gnp = random_gnp(n_nodes, p_lags, pole_rad=0.75,
                       edge_prob=edge_prob)

    draw_graph(G_tree, ["../figures/example_tree.pdf",
                        "../figures/jpgs_pngs/example_tree.png"])
    draw_graph(G_dag, ["../figures/example_dag.pdf",
                       "../figures/jpgs_pngs/example_dag.png"])
    draw_graph(G_gnp, ["../figures/example_gnp.pdf",
                       "../figures/jpgs_pngs/example_gnp.png"])
    return
