"""
Simulations with T held fixed and n increasing.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

from plotting_helpers import (COLOR3, COLOR4, COLOR5,
                              plotting_model)

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


def pwgc_increasing_n(T, simulation_name):
    np.random.seed(0)
    alpha, p_lags, p_max = 0.05, 5, 15
    T_max = 2 * T

    n_iters = np.arange(10, 1500, 10)
    N_reps = 2

    def random_graph_scg(n):
        return random_scg(n, p_lags, pole_rad=0.75)

    def random_graph_dag_q2(n):
        N_edges = (len(random_scg(n, p_lags,
                                  pole_rad=0.75).edges) -
                   n)  # Due to self loops
        edge_prob_q2 = 2 * N_edges / (n * (n - 1))
        return random_gnp_dag(n, p_lags, pole_rad=0.75,
                              edge_prob=edge_prob_q2)

    def random_graph_dag_q4(n):
        N_edges = (len(random_scg(n, p_lags,
                                  pole_rad=0.75).edges) -
                   n)  # Due to self loops
        edge_prob_q4 = 4 * N_edges / (n * (n - 1))
        return random_gnp_dag(n, p_lags, pole_rad=0.75,
                              edge_prob=edge_prob_q4)

    def make_test_data(n, graph_type):
        sv2_true = 0.5 + np.random.exponential(0.5, size=n)
        if graph_type == "SCG":
            G = random_graph_scg(n)
        elif graph_type == "DAG_q2":
            G = random_graph_dag_q2(n)
        elif graph_type == "DAG_q4":
            G = random_graph_dag_q4(n)

        drive_gcg(G, T_max, sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        X = X
        return G, X, sv2_true

    def get_results(graph_type):
        errs_pwgc = TrackErrors(n_iters, N_reps)
        for n_iter, n in enumerate(n_iters):
            for rep in range(N_reps):
                print("n[{} / {}]".format(n_iter + 1, len(n_iters)),
                      end="\r")

                G, X, sv2_true = make_test_data(n, graph_type)
                G_hat_pwgc = estimate_graph(X[:T], G, max_lags=p_max,
                                            method="lstsqr", alpha=alpha)
                errs_pwgc.update(G, G_hat_pwgc, sv2_true, n_iter, rep)

        mcc, _, _, _ = errs_pwgc.get_results()
        return mcc

    mcc_scg = get_results("SCG")
    mcc_dag_q2 = get_results("DAG_q2")
    mcc_dag_q4 = get_results("DAG_q4")

    _plot_results(
        n_iters, mcc_scg, mcc_dag_q2, mcc_dag_q4,
        title="Test Errors Against $n$ for $T = {}$".format(T),
        save_file=["../figures/{}_simulation.png".format(simulation_name),
                   "../figures/{}_simulation.pdf".format(simulation_name)])
    return


def _plot_results(n_iters, mcc_scg, mcc_dag_q2, mcc_dag_q4,
                  title=None, save_file=None):
    n_iters = np.vstack([n_iters] * mcc_scg.shape[1]).T
    n_iters, mcc_scg, mcc_dag_q2, mcc_dag_q4 = list(
        map(np.ravel, [n_iters, mcc_scg, mcc_dag_q2, mcc_dag_q4]))

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_ylim(0, 1.25)

    ax.scatter(n_iters, mcc_scg, color=COLOR3, marker="o", alpha=0.75)
    ax.scatter(n_iters, mcc_dag_q2, color=COLOR4, marker="^", alpha=0.75)
    ax.scatter(n_iters, mcc_dag_q4, color=COLOR5, marker="P", alpha=0.75)

    ax.plot(n_iters, plotting_model(n_iters, mcc_scg),
            color=COLOR3, linewidth=2)
    ax.plot(n_iters, plotting_model(n_iters, mcc_dag_q2),
            color=COLOR4, linewidth=2)
    ax.plot(n_iters, plotting_model(n_iters, mcc_dag_q4),
            color=COLOR5, linewidth=2)

    ax.set_xlabel("System size $n$")
    ax.set_ylabel("Graph Recovery MCC Score")

    ax.plot([], [], color=COLOR3, marker="o", label="SCG")
    ax.plot([], [], color=COLOR4, marker="^", label="DAG $q = \\frac{2}{n}$")
    ax.plot([], [], color=COLOR5, marker="P", label="DAG $q = \\frac{4}{n}$")
    ax.legend(loc="upper right")

    if save_file is not None:
        if isinstance(save_file, str):
            fig.savefig(save_file)
        else:  # Allow passing a list
            _ = [fig.savefig(f) for f in save_file]
    plt.show()
    return


if __name__ == "__main__":
    pwgc_increasing_n(T=500, simulation_name="new_increasing_n")
