"""
Simulation with both T and n held fixed, but with increasing
ER edge probability in the DAG generator.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

from plotting_helpers import (COLOR1, COLOR2, plotting_model)

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
    def __init__(self, q_iters, N_reps, n):
        self.q_iters = q_iters
        self.n = n
        self.N_reps = N_reps
        self.b_errs = np.zeros((len(self.q_iters), N_reps))
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
            self.N_cross_intersect_edges, self.n)

        return MCC, self.b_errs, self.errs, self.true_errs


def pwgc_increasing_q(T):
    simulation_name = "dag_increasing_q_small_T_alasso_pmax15"

    np.random.seed(0)
    alpha, p_lags, p_max, T = 0.05, 5, 15, 500
    T_max = 2 * T

    n = 50
    edge_probs = np.linspace(2. / n**2, 1. / np.log(n), 200)
    N_reps = 2

    def random_graph(q):
        return random_gnp_dag(n, p_lags, pole_rad=0.75,
                              edge_prob=q)
        # return random_scg(n, p_lags, pole_rad=0.75)

    def make_test_data(q):
        sv2_true = 0.5 + np.random.exponential(0.5, size=n)
        G = random_graph(q)
        drive_gcg(G, T_max, sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        X = X
        return G, X, sv2_true

    errs_pwgc = TrackErrors(edge_probs, N_reps, n)
    errs_lasso = TrackErrors(edge_probs, N_reps, n)

    for q_iter, q in enumerate(edge_probs):
        for rep in range(N_reps):
            print("q[{} / {}]".format(q_iter + 1, len(edge_probs)),
                  end="\r")

            G, X, sv2_true = make_test_data(q)
            G_hat_pwgc = estimate_graph(X[:T], G, max_lags=p_max,
                                        method="lstsqr", alpha=alpha)
            G_hat_lasso = estimate_dense_graph(X, max_lag=p_lags,
                                               max_T=T, method="alasso")
            errs_pwgc.update(G, G_hat_pwgc, sv2_true, q_iter, rep)
            errs_lasso.update(G, G_hat_lasso, sv2_true, q_iter, rep)

    mcc_pwgc, _, errs_pwgc_vec, true_errs = errs_pwgc.get_results()
    mcc_pwgc[np.isnan(mcc_pwgc)] = 1.0  # There is only 1

    mcc_lasso, _, errs_lasso_vec, _ = errs_lasso.get_results()
    mcc_lasso[np.isnan(mcc_lasso)] = 4e-2  # There is only 1

    _plot_results(
        edge_probs, mcc_pwgc, errs_pwgc_vec, mcc_lasso, errs_lasso_vec, true_errs,
        title="Test Errors Against $q$ for $T = {}, n = {}$".format(T, n),
        save_file=["../figures/{}_simulation.png".format(simulation_name),
                   "../figures/{}_simulation.pdf".format(simulation_name)])
    return


def _plot_results(q_iters, mcc_pwgc, errs_pwgc,
                  mcc_lasso, errs_lasso,
                  true_errs, title=None, save_file=None):

    def _plot_single(var_pwgc, var_lasso, ax, title, ylabel):
        ax.set_title(title)

        ax.set_xlabel("Erdos-Renyi Edge probability")
        ax.set_ylabel(ylabel)

        loc_q_iters = np.vstack([q_iters] * var_pwgc.shape[1]).T.ravel()

        loc_q_iters, var_pwgc, var_lasso = list(
            map(np.ravel, [loc_q_iters, var_pwgc, var_lasso]))

        ax.scatter(loc_q_iters, var_pwgc, color=COLOR1, marker="o",
                   alpha=0.75)
        ax.scatter(loc_q_iters, var_lasso, color=COLOR2, marker="^",
                        alpha=0.75)

        ax.plot(loc_q_iters, plotting_model(loc_q_iters, var_pwgc),
                color=COLOR1, linewidth=2)
        ax.plot(loc_q_iters, plotting_model(
            loc_q_iters, var_lasso),
                color=COLOR2, linewidth=2)
        return

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)

    log_normed_errs_pwgc = np.log(errs_pwgc) / np.log(true_errs)
    log_normed_errs_lasso = np.log(errs_lasso) / np.log(true_errs)

    _plot_single(mcc_pwgc, mcc_lasso, axes[0], title="MCC Comparison", ylabel="MCC")
    _plot_single(log_normed_errs_pwgc, log_normed_errs_lasso, axes[1],
                 title="Prediction Error Comparison",
                 ylabel="Log Prediction Error Relative to Noise Floor")
    fig.suptitle(title)

    axes[0].set_ylim(0, 1.25)

    axes[0].plot([], [], color=COLOR1, marker="o", label="PWGC")
    axes[0].plot([], [], color=COLOR2, marker="^", label="Adaptive LASSO")
    axes[0].legend(loc="upper left")

    if save_file is not None:
        if isinstance(save_file, str):
            fig.savefig(save_file)
        else:  # Allow passing a list
            _ = [fig.savefig(f) for f in save_file]
    plt.show()
    return


if __name__ == "__main__":
    pwgc_increasing_q(500)
