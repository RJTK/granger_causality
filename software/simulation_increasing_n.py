"""
Simulations with T held fixed and n increasing.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

from matplotlib import rc as mpl_rc
font = {"family" : "normal",
        "weight" : "bold",
        "size"   : 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=True)


from pwgc.var_system import (random_gnp_dag, drive_gcg, get_errors,
                             get_estimation_errors, random_scg,
                             get_X, attach_node_prop, gcg_to_var)
from pwgc.gc_methods import (compute_pairwise_gc, estimate_B, pw_scg,
                             estimate_graph, full_filter_estimator,
                             compute_MCC_score, estimate_dense_graph)


class TrackErrors:
    def __init__(self, n_iters, N_reps):
        self.n_iters = n_iters
        self.N_reps = N_reps
        self.b_errs = np.zeros((len(self.n_iters), N_reps))
        self.errs = np.zeros(self.b_errs.shape)
        self.true_errs = np.zeros(self.b_errs.shape)
        self.N_cross_edges = np.zeros(self.b_errs.shape)
        self.N_hat_cross_edges = np.zeros(self.b_errs.shape)
        self.N_cross_intersect_edges = np.zeros(self.b_errs.shape)
        return

    def update(self, G, G_hat, sv2_true, n_iter, rep):
        b_err = np.mean(list(map(np.var,
                                 get_estimation_errors(G, G_hat))))
        err = np.sum(get_errors(G_hat))
        true_err = np.sum(sv2_true)

        self.b_errs[n_iter, rep] = b_err
        self.errs[n_iter, rep] = err
        self.true_errs[n_iter, rep] = true_err

        self.N_cross_edges[n_iter, rep] = len(
            set((i, j) for i, j in G.edges if i != j))
        self.N_hat_cross_edges[n_iter, rep] = len(set(
            (i, j) for i, j in G_hat.edges if i != j))
        self.N_cross_intersect_edges[n_iter, rep] = len(
            (set(G.edges) & set(G_hat.edges)) - set((i, i)
                                                    for i in G.nodes))
        return

    def get_results(self):
        MCC = compute_MCC_score(
            self.N_cross_edges, self.N_hat_cross_edges,
            self.N_cross_intersect_edges, self.n_iters[:, None])

        return MCC, self.b_errs, self.errs, self.true_errs


def pwgc_increasing_n(T, simulation_name, graph_type):
    np.random.seed(0)
    alpha, p_lags, p_max = 0.05, 5, 15
    T_max = 2 * T

    n_iters = np.arange(10, 2000, 20)
    N_reps = 3

    if graph_type == "SCG":
        def random_graph(n):
            return random_scg(n, p_lags, pole_rad=0.75)
    elif graph_type == "DAG":
        def random_graph(n):
            N_edges = (len(random_scg(n, p_lags,
                                      pole_rad=0.75).edges) -
                       n)  # Due to self loops
            edge_prob = 2 * N_edges / (n * (n - 1))
            return random_gnp_dag(n, p_lags, pole_rad=0.75,
                                  edge_prob=edge_prob)
    else:
        raise NotImplementedError("Only 'SCG' and 'DAG' graph "
                                  "types are available")

    def make_test_data(n):
        sv2_true = 0.5 + np.random.exponential(0.5, size=n)
        G = random_graph(n)
        drive_gcg(G, T_max, sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        X = X
        return G, X, sv2_true

    errs_pwgc = TrackErrors(n_iters, N_reps)

    for n_iter, n in enumerate(n_iters):
        for rep in range(N_reps):
            print("n[{} / {}]\r".format(n_iter + 1, len(n_iters)))

            G, X, sv2_true = make_test_data(n)
            G_hat_pwgc = estimate_graph(X[:T], G, max_lags=p_max,
                                        method="lstsqr", alpha=alpha)
            errs_pwgc.update(G, G_hat_pwgc, sv2_true, n_iter, rep)

    mcc, _, errs, true_errs = errs_pwgc.get_results()
    _plot_results(
        n_iters[:-22], mcc[:-22], errs[:-22], true_errs[:-22],
        title="Test Errors Against $n$ for $T = {}$".format(T),
        save_file=["../figures/{}_simulation.png".format(simulation_name),
                   "../figures/{}_simulation.pdf".format(simulation_name)])
    return


def _plot_results(n_iters, mcc, errs, true_errs,
                  title=None, save_file=None):
    log_normed_errs = np.log(errs) / np.log(true_errs)
    n_iters = np.vstack([n_iters] * mcc.shape[1]).T
    n_iters, mcc, log_normed_errs = list(map(np.ravel,
                                             [n_iters, mcc, log_normed_errs]))

    fig, ax_mcc = plt.subplots(1, 1)
    fig.suptitle(title)

    ax_pred = ax_mcc.twinx()
    ax_pred.grid(False)

    ax_mcc.scatter(n_iters, mcc, color="b", marker="o")
    ax_pred.scatter(n_iters, log_normed_errs, color="r", marker="^")

    ax_mcc.plot(n_iters, _fit_plotting_model(n_iters, mcc),
                color="b", linewidth=2)
    ax_pred.plot(n_iters, _fit_plotting_model(n_iters, log_normed_errs),
                color="r", linewidth=2)

    ax_mcc.set_title("PWGC")

    ax_mcc.set_xlabel("System size $n$")
    ax_mcc.set_ylabel("Graph Recovery MCC Score")
    ax_pred.set_ylabel("Log Prediction Error Relative to Noise Floor")

    ax_mcc.plot([], [], color="b", marker="o", label="MCC")
    ax_mcc.plot([], [], color="r", marker="^", label="Prediction Error")
    ax_mcc.legend(loc="upper left")

    if save_file is not None:
        if isinstance(save_file, str):
            fig.savefig(save_file)
        else:  # Allow passing a list
            _ = [fig.savefig(f) for f in save_file]
    plt.show()
    return


def _fit_plotting_model(x, y):
    X = np.vstack((np.ones_like(x), x, np.sqrt(x))).T

    fit = sm.OLS(endog=y, exog=X, hasconst=True)
    fit_res = fit.fit()
    y_hat = fit_res.fittedvalues
    return y_hat
