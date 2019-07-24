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
    n_nodes, p_lags, p_max = 35, 5, 10
    alpha = 0.05

    T_max = 1000

    random_graph = lambda: random_scg(
        n_nodes, p_lags, pole_rad=0.75)
    # random_graph = lambda: random_gnp_dag(
    #     n_nodes, p_lags, pole_rad=0.75, edge_prob=4./n_nodes)
    sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
    G = random_graph()
    drive_gcg(G, T_max, sv2_true, filter_attr="b(z)")
    X = get_X(G)
    X = X - np.mean(X, axis=0)[None, :]

    def est_graph(X, fast=True, bd=False, F_distr=False):
        if fast:
            F, P = fast_compute_pairwise_gc(X, p_lags)
        else:
            F, P = compute_pairwise_gc(X, max_lag=p_lags)

        # p-values are just 1 - F
        P_edges = normalize_gc_score(F, P, T=len(X), F_distr=F_distr)
        P_values = 1 - P_edges[~np.eye(n_nodes, dtype=bool)].ravel()
        t_bh = benjamini_hochberg(P_values, alpha=alpha, independent=False)
        G_hat = pw_scg(F, P_edges, t_bh, eliminate_bidirectional=False)
        draw_graph_estimates(G, G_hat)
        return

    # import networkx as nx
    # def est_graph(X, fast=True, bd=False):
    #     if fast:
    #         F, P = fast_compute_pairwise_gc(X, p_lags)
    #     else:
    #         F, P = compute_pairwise_gc(X, max_lag=p_lags)

    #     # p-values are just 1 - F
    #     P_edges = normalize_gc_score(F, P, T=len(X), F_distr=False)
    #     P_values = 1 - P_edges[~np.eye(n_nodes, dtype=bool)].ravel()
    #     t_bh = benjamini_hochberg(P_values, alpha=alpha, independent=False)
    #     P_edges[P_edges < 1 - t_bh] = 0
    #     P_edges[P_edges > 0] = 1
    #     G_hat = nx.DiGraph(P_edges.T)
    #     # G_hat = pw_scg(F, P_edges, t_bh, eliminate_bidirectional=bd)
    #     draw_graph_estimates(G, G_hat)
    #     return

    # est_graph(X, fast=False)
    # est_graph(X, fast=True)
    # est_graph(X, bd=False, F_distr=True)
    # est_graph(X, bd=False, F_distr=False)
    est_graph(X, bd=True, F_distr=False)
    est_graph(X, bd=False, F_distr=False)
    return


def stat_sig():
    N = 50000
    N_iter = 50

    def make_ar(p=0.9):
        a = np.zeros(N)
        v = np.random.normal(size=N)
        if p:
            a[0] = (1 / (1 - p)) * np.random.normal()
            for t in range(N - 1):
                a[t + 1] = p * a[t] + v[t]
            return a
        else:
            return v

    F, P = np.zeros((2, 2)), np.zeros((2, 2))
    for _ in range(N_iter):
        a = make_ar(0.9)
        b = make_ar(0.9)
        X = np.vstack((a, b)).T

        # xi_i, xi_ij, p_i, p_ij = fast_compute_xi(np.vstack((a, b)).T, max_lag=2)
        _F, p_ij = fast_compute_pairwise_gc(X)
        # _F = compute_gc_score(xi_i, xi_ij, T=N, p_lags=p_ij, F_distr=False)
        _P = normalize_gc_score(F, p_ij, T=N, F_distr=False)
        F += _F
        P += _P
    P = P / N_iter
    F = F / N_iter
    print(P)
    print(F)
# test_pw_scg()
