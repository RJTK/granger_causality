"""
Generates some example graph topologies to include in paper.
"""

from pwgc.var_system import (random_scg, random_gnp_dag, random_gnp)
from pwgc.draw_graphs import draw_graph


def draw_example_graphs():
    n_nodes, p_lags = 75, 5
    edge_prob = 2. / n_nodes  # Set to 2. / n_nodes for same edges as G_scg

    G_scg = random_scg(n_nodes, p_lags, pole_rad=0.75)
    G_dag = random_gnp_dag(n_nodes, p_lags, pole_rad=0.75,
                           edge_prob=edge_prob)
    G_dag_q = random_gnp_dag(n_nodes, p_lags, pole_rad=0.75,
                                  edge_prob=2 * edge_prob)
    G_gnp = random_gnp(n_nodes, p_lags, pole_rad=0.75,
                       edge_prob=edge_prob)

    draw_graph(G_scg, ["../figures/example_scg.pdf",
                        "../figures/jpgs_pngs/example_tree.png"])
    draw_graph(G_dag, ["../figures/example_dag.pdf",
                       "../figures/jpgs_pngs/example_dag.png"])
    draw_graph(G_dag_q, ["../figures/example_dag_q.pdf",
                         "../figures/jpgs_pngs/example_dag_q.png"])
    draw_graph(G_gnp, ["../figures/example_gnp.pdf",
                       "../figures/jpgs_pngs/example_gnp.png"])
    return


if __name__ == "__main__":
    draw_example_graphs()
