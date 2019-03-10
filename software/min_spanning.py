import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout


def complete_digraph(n=10, wgt_f=lambda: np.random.standard_t(df=3)):
    G = nx.DiGraph()
    for i in range(n):
        [G.add_edge(i, j, weight=wgt_f()) for j in range(n)
         if j != i]
    return G


G = complete_digraph(n=25,
                     wgt_f=lambda: np.random.standard_t(df=1))

print("Searching for spanning arborescence")
T = nx.minimum_spanning_arborescence(G, attr="weight",
                                     default=np.inf)
print("Done.")

pos = graphviz_layout(G, prog="dot")
nx.draw(T, pos=pos, node_color="b", edgelist=list(T.edges),
        edge_color=[T.get_edge_data(u, v)["weight"] for u, v in T.edges],
        width=2, edge_cmap=plt.cm.viridis)
# plt.colorbar()
plt.show()
