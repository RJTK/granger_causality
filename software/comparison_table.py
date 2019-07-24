"""
Create a condensed table of data for comparison against AdaLASSO.
"""

import pandas as pd
import numpy as np
import xarray as xr

from itertools import product

from scipy.stats import ttest_rel
from pwgc.var_system import (random_gnp_dag, drive_gcg, get_errors,
                             get_estimation_errors, random_scg,
                             get_X, attach_node_prop, gcg_to_var)
from pwgc.gc_methods import (compute_pairwise_gc, estimate_B, pw_scg,
                             estimate_graph, full_filter_estimator,
                             compute_MCC_score, estimate_dense_graph,
                             alasso_fista_estimate_dense_graph,
                             compute_fdr_score)

alpha = 0.05


def main():
    np.random.seed(0)
    n_nodes, p_lags, p_max = 50, 5, 10
    N_iters = 100

    params = {"T": [50, 250, 1250],
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
        for it in range(N_iters):
            G = make_random_graph(q)
            X, sv2_true = make_test_data(T, G)

            G_hat_lasso = adalasso(X, G, T, p_max)
            G_hat_pwgc = pwgc(X, G, T, p_max)
            results["alasso"].loc[T, str(q), it, :] = _create_result_da(
                *calculate_error(G_hat_lasso, G, sv2_true))
            results["pwgc"].loc[T, str(q), it, :] = _create_result_da(
                *calculate_error(G_hat_pwgc, G, sv2_true))

    table = results_to_latex_table(results, params)
    print(table)
    return


def results_to_latex_table(results_xr, params):
    def _boldify(s):
        return "\textbf{{{}}}".format(s)

    a_level = 0.05

    # Perform a ttest for related data over the iteration axis
    significance_tests = ttest_rel(results_xr["alasso"].values,
                                   results_xr["pwgc"].values,
                                   axis=2)[1] < a_level

    D_sig = results_xr["alasso"].mean("iter")\
        .copy()
    D_sig.values = significance_tests
    D_sig = D_sig.to_dataframe()\
        .reset_index()\
        .pivot_table(index=["T", "q"], columns="metric")\
        .sort_index(axis=0, level=[0, 1], ascending=[True, True])\
        .sort_index(axis=1, level=1)

    D_mu = results_xr.mean("iter")\
        .to_dataframe()\
        .reset_index()\
        .pivot_table(index=["T", "q"], columns="metric")\
        .sort_index(axis=0, level=[0, 1], ascending=[True, True])\
        .sort_index(axis=1, level=1)

    D_mu = D_mu.astype(str)

    # This is a messy hack to properly mark significance
    for i in range(len(D_sig.values)):
        for j in range(len(D_sig.values[i, :])):
            mu_alasso = float(D_mu.values[i, 2 * j])
            mu_pwgc = float(D_mu.values[i, 2 * j + 1])

            D_mu.values[i, 2 * j] = fltfmt(mu_alasso)
            D_mu.values[i, 2 * j + 1] = fltfmt(mu_pwgc)

            if D_sig.columns.levels[1][j] == "MCC":
                big_better = True
            else:
                big_better = False

            if D_sig.values[i, j]:
                if (((mu_alasso > mu_pwgc) and big_better) or
                        (mu_alasso < mu_pwgc) and not big_better):
                    D_mu.values[i, 2 * j] = _boldify(
                        D_mu.values[i, 2 * j])
                else:
                    D_mu.values[i, 2 * j + 1] = _boldify(
                        D_mu.values[i, 2 * j + 1])

    table = D_mu.to_latex(bold_rows=True, escape=False,
                          multicolumn=True, multirow=True,
                          float_format="%0.3f")
    for qi in params["q"]:
        if qi != "SCG":
            table = table.replace(str(qi), fltfmt(qi))
    return table


def calculate_error(G_hat, G_true, sv2_true):
    n_nodes = len(G_hat.nodes)

    true_err = np.log(np.sum(sv2_true))
    est_err = np.log(np.sum(get_errors(G_hat)))
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

    fdr = compute_fdr_score(N_cross_edges, N_hat_cross_edges,
                            N_cross_intersect_edges, n_nodes)
    return mcc, err, fdr


def adalasso(X, G, T, p_max):
    return estimate_dense_graph(X, max_lag=p_max, max_T=T,
                                method="alasso", post_ols=True)


def pwgc(X, G, T, p_max):
    return estimate_graph(X[:T], G, max_lags=p_max,
                          method="lstsqr", alpha=alpha,
                          fast_mode=True, F_distr=False)


def fltfmt(f):
    return "{:0.2f}".format(f)
