"""
Create a condensed table of data for comparison against AdaLASSO.
"""

import pandas as pd
import numpy as np
import xarray as xr

from itertools import product

from pwgc.var_system import (random_gnp_dag, drive_gcg, get_errors,
                             get_estimation_errors, random_scg,
                             get_X, attach_node_prop, gcg_to_var)
from pwgc.gc_methods import (compute_pairwise_gc, estimate_B, pw_scg,
                             estimate_graph, full_filter_estimator,
                             compute_MCC_score, estimate_dense_graph,
                             alasso_fista_estimate_dense_graph, compute_fdr_score)

alpha = 0.05


def main():
    np.random.seed(0)
    n_nodes, p_lags, p_max = 5, 5, 10
    N_iters = 5

    params = {"T": [50, 250, 1000],
              "q": ["SCG", 2. / n_nodes, 4. / n_nodes, 16. / n_nodes],
              "metric": ["MCC", "Err", "FDR"],
              "iter": list(range(N_iters))}

    def create_empty_ds():
        _adalasso = _create_empty_da()
        _pwgc = _create_empty_da()
        return xr.Dataset(coords=params,
                          data_vars={"alasso": _adalasso,
                                     "pwgc": _pwgc})

    def _create_empty_da():
        return xr.DataArray(data=np.nan * np.empty((len(params["T"]),
                                                    len(params["q"]),
                                                    N_iters,
                                                    len(params["metric"]))),
                            dims=["T", "q", "iter", "metric"],
                            coords=params)

    def _create_result_da(mcc, err, fdr):
        return xr.DataArray(data=np.array([mcc, err, fdr]),
                            dims=["metric"],
                            coords={"metric": ["MCC", "Err", "FDR"]})

    def make_random_graph(q):
        if q == "SCG":
            return random_scg(n_nodes, p_lags, pole_rad=0.75)
        else:
            return random_gnp_dag(n_nodes, p_lags, pole_rad=0.75,
                                  edge_prob=q)

    def make_test_data(T, G):
        sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
        drive_gcg(G, 500 + min(2 * T, T + 15000), sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        X = X
        return X, sv2_true

    results = create_empty_ds()
    for T, q in product(params["T"], params["q"]):
        G = make_random_graph(q)
        X, sv2_true = make_test_data(T, G)

        for it in range(N_iters):
            G_hat_lasso = adalasso(X, G, T, p_max)
            G_hat_pwgc = pwgc(X, G, T, p_max)
            results["alasso"].loc[T, str(q), it, :] = _create_result_da(
                *calculate_error(G_hat_lasso, G, sv2_true))
            results["pwgc"].loc[T, str(q), it, :] = _create_result_da(
                *calculate_error(G_hat_lasso, G, sv2_true))
    return


def calculate_error(G_hat, G_true, sv2_true):
    n_nodes = len(G_hat.nodes)

    true_err = np.sum(sv2_true)
    est_err = np.sum(get_errors(G_hat))
    err = est_err / true_err

    def _compute_cross_edges(G):
        return len(set((i, j) for i, j in G.edges if i != j))

    N_hat_cross_edges = _compute_cross_edges(G_hat)
    N_cross_edges = _compute_cross_edges(G_true)
    N_cross_intersect_edges = len(
        (set(G_true.edges) & set(G_hat.edges)) -
        set((i, i) for i in G_true.nodes))
    mcc = compute_MCC_score(N_cross_edges, N_hat_cross_edges,
                            N_cross_intersect_edges, n_nodes)

    fdr = compute_fdr_score(None)
    return mcc, err, fdr


def adalasso(X, G, T, p_max):
    return estimate_dense_graph(X, max_lag=p_max, max_T=T, method="alasso")


def pwgc(X, G, T, p_max):
    return estimate_graph(X[:T], G, max_lags=p_max,
                          method="lstsqr", alpha=alpha,
                          fast_mode=True, F_distr=True)
