import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc as mpl_rc
font = {"family": "normal",
        "weight": "bold",
        "size": 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=False)

from var_system import (random_gnp_dag, drive_gcg, get_errors,
                        get_estimation_errors, random_scg,
                        get_X, attach_node_prop, gcg_to_var)
from gc_methods import (compute_pairwise_gc, estimate_B, pw_scg,
                        estimate_graph, full_filter_estimator,
                        compute_covariances, compute_xi, fast_compute_xi,
                        normalize_gc_score, fast_compute_pairwise_gc)
from draw_graphs import draw_graph_estimates
from stat_util import benjamini_hochberg


def test_fast_computation():
    """
    Test the "fast" mode computations.

    TODO: They appear to still produce terrible results for the
    TODO: complete simulations even though the results are similar
.   TODO: here.  The covariance estimates are frequently indefinite.
    """
    n_nodes, p_lags, max_lags = 75, 5, 10
    random_graph = lambda: random_tree_dag(
        n_nodes, p_lags, pole_rad=0.25)

    T = 500
    sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
    G = random_graph()
    drive_gcg(G, T, sv2_true, filter_attr="b(z)")
    X = get_X(G)

    xi_i_corr, xi_ij_corr, _, p_ij = compute_xi(X, max_lag=max_lags)

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

    for delta in [0, 1e-3, 1e-2]:
        xi_i_delta, xi_ij_delta = fast_compute_xi(X, max_lag=max_lags,
                                                  reg_delta=delta)

        axes[0].scatter(xi_i_corr, xi_i_delta,
                        label="delta = {}".format(delta))
        axes[0].set_title("xi_i")

        axes[1].scatter(xi_ij_corr, xi_ij_delta,
                        label="delta = {}".format(delta))
        axes[1].set_title("xi_ij")

    axes[0].legend()
    t = np.linspace(0, np.max(xi_i_corr), 1000)
    [ax.plot(t, t) for ax in axes]
    [ax.set_ylabel("Fast Values") for ax in axes]
    [ax.grid(True) for ax in axes]
    plt.tight_layout()
    plt.show()
    return


def test_pw_scg():
    n_nodes, p_lags, p_max = 50, 5, 15
    alpha = 0.05

    T_max = 5000

    random_graph = lambda: random_scg(
        n_nodes, p_lags, pole_rad=0.75)
    # random_graph = lambda: random_gnp_dag(
    #     n_nodes, p_lags, pole_rad=0.75, edge_prob=2./n_nodes)
    sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
    G = random_graph()
    drive_gcg(G, T_max, sv2_true, filter_attr="b(z)")
    X = get_X(G)
    X = X - np.mean(X, axis=0)[None, :]

    def est_graph(X, fast=False):
        if fast:
            F, P = fast_compute_pairwise_gc(X, p_lags)
        else:
            F, P = compute_pairwise_gc(X, max_lag=p_lags)

        P_edges = normalize_gc_score(F, P)  # p-values are just 1 - F
        P_values = 1 - P_edges[~np.eye(n_nodes, dtype=bool)].ravel()
        t_bh = benjamini_hochberg(P_values, alpha=alpha, independent=False)
        G_hat = pw_scg(F, P_edges, t_bh)
        draw_graph_estimates(G, G_hat)
        return

    est_graph(X, False)
    est_graph(X, True)
    return


# test_pw_scg()
