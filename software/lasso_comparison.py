"""
Comparisons for PWGC and LASSO.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


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
    def __init__(self, N_iters, T_iters, n_nodes):
        self.n_nodes = n_nodes
        self.T_iters = T_iters
        N_T_iters = len(T_iters)

        self.b_errs = np.zeros((N_iters, N_T_iters))
        self.errs = np.zeros(self.b_errs.shape)
        self.true_errs = np.zeros(self.b_errs.shape)
        self.rhos = np.zeros(self.b_errs.shape)
        self.N_cross_edges = np.zeros(self.b_errs.shape)
        self.N_hat_cross_edges = np.zeros(self.b_errs.shape)
        self.N_cross_intersect_edges = np.zeros(self.b_errs.shape)
        return

    def update(self, G, G_hat, sv2_true, N_iter, T_iter):
        b_err = np.mean(list(map(np.var,
                                 get_estimation_errors(G, G_hat))))
        err = np.sum(get_errors(G_hat))
        true_err = np.sum(sv2_true)

        self.rhos[N_iter, T_iter] = gcg_to_var(
            G_hat, "b_hat(z)", assert_stable=False).get_rho()

        self.b_errs[N_iter, T_iter] = b_err
        self.errs[N_iter, T_iter] = err
        self.true_errs[N_iter, T_iter] = true_err
        self.N_cross_edges[N_iter, T_iter] = len(
            set((i, j) for i, j in G.edges if i != j))
        self.N_hat_cross_edges[N_iter, T_iter] = len(set(
            (i, j) for i, j in G_hat.edges if i != j))
        self.N_cross_intersect_edges[N_iter, T_iter] = len(
            (set(G.edges) & set(G_hat.edges)) - set((i, i)
                                                    for i in G.nodes))
        return

    def get_results(self):
        MCC = compute_MCC_score(
            self.N_cross_edges, self.N_hat_cross_edges,
            self.N_cross_intersect_edges, self.n_nodes)

        def to_pd(X):
            D = pd.DataFrame(
                X, columns=["{}".format(T) for T in self.T_iters]).melt()
            return D

        D_errs = to_pd(np.log(self.errs))
        D_true_errs = to_pd(np.log(self.true_errs))
        D_MCC = to_pd(MCC)
        return D_errs, D_true_errs, D_MCC


def lasso_comparison(simulation_name, graph_type):
    np.random.seed(0)
    n_nodes, p_lags, p_max = 50, 5, 15
    alpha, N_iters = 0.05, 3

    # T_iters = list(map(int, np.logspace(2, 3, 20)))
    # T_iters = list(map(int, np.linspace(30, 5000, 20)))
    # T_iters = [50 + 100 * k for k in range(1, 99)]
    T_iters = [50 + 100 * k for k in range(1, 99, 1)]
    # T_iters = [20, 40, 80, 100, 150, 250, 500]

    # NOTE: We use only X[T:] to estimate the error
    # NOTE: so it is important to have a significant trailing
    # NOTE: set of data in order to make honest estimates of
    # NOTE: the out of sample predictions.

    # Tune the expected number of edges to match N_edges
    N_edges = (len(random_scg(n_nodes, p_lags, pole_rad=0.75).edges) -
               n_nodes)  # Due to self loops
    edge_prob = 2 * N_edges / (n_nodes * (n_nodes - 1))

    if graph_type == "SCG":
        random_graph = lambda: random_scg(
            n_nodes, p_lags, pole_rad=0.75)
    elif graph_type == "DAG":
        random_graph = lambda: random_gnp_dag(
            n_nodes, p_lags, pole_rad=0.75,
            edge_prob=edge_prob)
    elif graph_type == "ErdosRenyi":
        # Tune the expected number of edges to match N_edges
        random_graph = lambda: random_gnp(
            n_nodes, p_lags, pole_rad=0.75,
            edge_prob=edge_prob)
    else:
        raise NotImplementedError("graph_type '{}' is not understood"
                                  "".format(graph_type))

    def make_test_data(T):
        sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
        G = random_graph()
        drive_gcg(G, 500 + 2 * T, sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        X = X
        return G, X, sv2_true

    errs_pwgc = TrackErrors(N_iters, T_iters, n_nodes)
    errs_lasso = TrackErrors(N_iters, T_iters, n_nodes)

    for T_iter, T in enumerate(T_iters):
        for N_iter, _ in enumerate(range(N_iters)):
            print("N[{} / {}] -- T[{} / {}]\r".format(
                N_iter + 1, N_iters, T_iter + 1, len(T_iters)))

            # ~252ms
            G, X, sv2_true = make_test_data(T)

            # ~824ms
            # TODO: F_distr is incorrect
            G_hat_pwgc = estimate_graph(X[:T], G, max_lags=p_max,
                                        method="lstsqr", alpha=alpha,
                                        fast_mode=True, F_distr=False)
            # G_hat_lasso = estimate_graph(X[:T], G, max_lags=p_max,
            #                             method="lstsqr", alpha=alpha,
            #                             fast_mode=True, F_distr=True)

            # ~37s
            # G_hat_lasso = estimate_dense_graph(X, max_lag=p_max,
            #                                    max_T=T,
            #                                    method="lasso")
            # G_hat_lasso = estimate_dense_graph(X, max_lag=p_max,
            #                                    max_T=T,
            #                                    method="glasso")
            G_hat_lasso = estimate_dense_graph(X, max_lag=p_max,
                                               max_T=T,
                                               method="alasso")

            # G_hat_lasso = estimate_dense_graph(X, max_lag=p_max,
            #                                    max_T=T,
            #                                    method="alasso")
            # G_hat_lasso = estimate_dense_graph(X, max_lag=p_max,
            #                                    max_T=T,
            #                                    method="lasso")

            errs_pwgc.update(G, G_hat_pwgc, sv2_true, N_iter, T_iter)
            errs_lasso.update(G, G_hat_lasso, sv2_true, N_iter, T_iter)

    D_errs_pwgc, D_true_errs, D_MCC_pwgc = errs_pwgc.get_results()
    D_errs_lasso, _, D_MCC_lasso = errs_lasso.get_results()

    plot_results(
        D_MCC_pwgc, D_errs_pwgc, D_MCC_lasso, D_errs_lasso,
        D_true_errs,
        title=("Test Errors against T (Random {} graph on "
               "$n = {}$ nodes)".format(graph_type, n_nodes)),
        lasso_title="Adaptive LASSO",
        save_file=["../figures/{}_simulation.pdf".format(simulation_name),
                   "../figures/jpgs_pngs/"
                   "{}_simulation.png".format(simulation_name)],
        show_results=False)
    return


def _fit_sqrt_model(x, y):
    X = np.vstack((np.ones_like(x), x, np.sqrt(x))).T
    X = np.vstack((np.ones_like(x), np.sqrt(x), np.sqrt(x + np.median(x)))).T

    fit = sm.OLS(endog=y, exog=X, hasconst=True)
    fit_res = fit.fit()
    y_hat = fit_res.fittedvalues
    _, y_low, y_high = wls_prediction_std(fit_res, X, alpha=0.1)
    return y_low, y_hat, y_high


def _plot_sqrt_model(x, y, y_low, y_hat, y_high, ax,
                     color="b", marker="o",
                     fill_between=True):
    ax.scatter(x, y, color=color, marker=marker, alpha=0.75,
               linewidths=0.0, edgecolors=color)
    ax.plot(x, y_hat, color=color, linewidth=2)
    if fill_between:
        ax.fill_between(x, y_low, y_high,
                        color=color, alpha=0.5)
    return


def _plot_individual_results(t, mcc, normed_errs, ax_mcc, title="",
                             legend=True):
    ax_mcc.set_title(title)

    ax_pred = ax_mcc.twinx()
    _plot_sqrt_model(t, mcc, *_fit_sqrt_model(t, mcc),
                     ax=ax_mcc, color="b", marker="o", fill_between=False)
    _plot_sqrt_model(t, normed_errs, *_fit_sqrt_model(t, normed_errs),
                     ax=ax_pred, color="r", marker="^", fill_between=False)

    ax_mcc.set_xlabel("Time Points $T$")
    ax_mcc.set_ylabel("Graph Recovery MCC Score")
    ax_pred.set_ylabel("Log Prediction Error Relative to Noise Floor")
    ax_mcc.set_ylim(0, 1.25)
    ax_pred.set_ylim(0.5, 3)

    if legend:
        ax_pred.plot([], [], color="b", marker="o", label="MCC")
        ax_pred.plot([], [], color="r", marker="^", label="Prediction Error")
        ax_pred.legend(loc="upper left")
    ax_pred.grid(False)
    return


def plot_results(D_MCC_pwgc, D_errs_pwgc,
                 D_MCC_lasso, D_errs_lasso,
                 D_true_errs,
                 save_file=None, title="",
                 lasso_title="LASSO",
                 show_results=True):
    true_errs = np.array(D_true_errs["value"], dtype=float)
    t = np.array(D_true_errs["variable"], dtype=float)

    def _D_to_np(D_MCC, D_errs):
        mcc = np.array(D_MCC["value"], dtype=float)
        errs = np.array(D_errs["value"], dtype=float)
        errs = errs / true_errs  # Error relative to noise floor
        return mcc, errs

    mcc_pwgc, errs_pwgc = _D_to_np(D_MCC_pwgc, D_errs_pwgc)
    mcc_lasso, errs_lasso = _D_to_np(D_MCC_lasso, D_errs_lasso)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    _plot_individual_results(t, mcc_pwgc, errs_pwgc, axes[0],
                             title="PWGC Algorithm", legend=True)
    _plot_individual_results(t, mcc_lasso, errs_lasso, axes[1],
                             title=lasso_title, legend=False)
    fig.suptitle(title)

    if save_file is not None:
        if isinstance(save_file, str):
            fig.savefig(save_file)
        else:  # Allow passing a list
            _ = [fig.savefig(f) for f in save_file]
    if show_results:
        plt.show()
    return
