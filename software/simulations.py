import seaborn.apionly as sns
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
        self.errs_vec = np.zeros((N_iters, N_T_iters, n_nodes))
        self.true_errs = np.zeros(self.b_errs.shape)
        self.true_errs_vec = np.zeros(self.errs_vec.shape)
        self.rhos = np.zeros(self.b_errs.shape)
        self.N_cross_edges = np.zeros(self.b_errs.shape)
        self.N_hat_cross_edges = np.zeros(self.b_errs.shape)
        self.N_cross_intersect_edges = np.zeros(self.b_errs.shape)
        return

    def update(self, G, G_hat, sv2_true, N_iter, T_iter):
        b_err = np.mean(list(map(np.var,
                                 get_estimation_errors(G, G_hat))))
        err = np.sum(get_errors(G_hat))
        err_vec = np.sort(get_errors(G_hat))
        true_err = np.sum(sv2_true)
        true_err_vec = np.sort(sv2_true)

        self.rhos[N_iter, T_iter] = gcg_to_var(
            G_hat, "b_hat(z)", assert_stable=False).get_rho()

        self.b_errs[N_iter, T_iter] = b_err
        self.errs[N_iter, T_iter] = err
        self.errs_vec[N_iter, T_iter] = err_vec
        self.true_errs[N_iter, T_iter] = true_err
        self.true_errs_vec[N_iter, T_iter] = true_err_vec
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


def full_single_pass_experiment(simulation_name, graph_type):
    np.random.seed(0)
    n_nodes, p_lags, p_max = 50, 5, 20
    alpha, N_iters = 0.05, 3

    T_iters = [50 + 50 * k for k in range(1, 99)]

    # NOTE: We use only X[T:] to estimate the error
    # NOTE: so it is important to have a significant trailing
    # NOTE: set of data in order to make honest estimates of
    # NOTE: the out of sample predictions.
    T_max = T_iters[-1] * 2

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
        drive_gcg(G, T_max, sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        X = X[:T]
        return G, X, sv2_true

    errs_pwgc = TrackErrors(N_iters, T_iters, n_nodes)
    errs_lasso = TrackErrors(N_iters, T_iters, n_nodes)

    # TODO: There is a ton of repeated work going on all over
    for T_iter, T in enumerate(T_iters):
        for N_iter, _ in enumerate(range(N_iters)):
            print("N[{} / {}] -- T[{} / {}]\r".format(
                N_iter + 1, N_iters, T_iter + 1, len(T_iters)))

            G, X, sv2_true = make_test_data(T)
            G_hat_pwgc = estimate_graph(X, G, max_lags=p_max,
                                        method="lstsqr", alpha=alpha)
            G_hat_lasso = estimate_dense_graph(X, max_lag=p_max,
                                               max_T=T,
                                               method="lasso")

            errs_pwgc.update(G, G_hat_pwgc, sv2_true, N_iter, T_iter)
            errs_lasso.update(G, G_hat_lasso, sv2_true, N_iter, T_iter)

    D_errs_pwgc, D_true_errs_pwgc, D_MCC_pwgc = errs_pwgc.get_results()
    D_errs_lasso, D_true_errs_lasso, D_MCC_lasso = errs_lasso.get_results()

    plot_results(
        D_MCC_pwgc, D_errs_pwgc, D_true_errs_pwgc, T_iters,
        title=("Test Errors against T (Random {} graph on "
               "$n = {}$ nodes)".format(graph_type, n_nodes)),
        save_file=["../figures/{}_simulation.pdf".format(simulation_name),
                   "../figures/jpgs_pngs/"
                   "{}_simulation.png".format(simulation_name)])
    return


def _fit_sqrt_model(x, y):
    X = np.vstack((np.ones_like(x), x, np.sqrt(x))).T
    X = np.vstack((np.ones_like(x), np.sqrt(x), np.sqrt(x + np.median(x)))).T

    fit = sm.OLS(endog=y, exog=X, hasconst=True)
    fit_res = fit.fit()
    y_hat = fit_res.fittedvalues
    _, y_low, y_high = wls_prediction_std(fit_res, X, alpha=0.1)
    return y_low, y_hat, y_high


def _plot_sqrt_model(x, y, y_low, y_hat, y_high, ax, color="b",
                     fill_between=True):
    ax.scatter(x, y, color=color, marker="o")
    ax.plot(x, y_hat, color=color, linewidth=2)
    if fill_between:
        ax.fill_between(x, y_low, y_high,
                        color=color, alpha=0.5)
    return


def plot_results(D_MCC, D_errs, D_true_errs, T_iters,
                 save_file=None, title=""):
    t = np.array(D_MCC["variable"], dtype=float)
    mcc = np.array(D_MCC["value"], dtype=float)
    errs = np.array(D_errs["value"], dtype=float)
    true_errs = np.array(D_true_errs["value"], dtype=float)

    errs = errs / true_errs  # Error relative to noise floor

    fig, ax_mcc = plt.subplots(1, 1)
    ax_pred = ax_mcc.twinx()
    order = ["{}".format(T) for T in T_iters]

    _plot_sqrt_model(t, mcc, *_fit_sqrt_model(t, mcc),
                     ax=ax_mcc, color="b", fill_between=False)
    _plot_sqrt_model(t, errs, *np.exp(_fit_sqrt_model(t, np.log(errs))),
                     ax=ax_pred, color="r", fill_between=False)

    ax_mcc.set_xlabel("Time Points $T$")
    ax_mcc.set_ylabel("Graph Recovery MCC Score")
    ax_pred.set_ylabel("Log Prediction Error Relative to Noise Floor")
    ax_pred.plot([], [], color="b", label="MCC")
    ax_pred.plot([], [], color="r", label="Prediction Error")
    ax_pred.legend(loc="upper left")
    ax_pred.grid(False)
    ax_mcc.set_title(title)

    if save_file is not None:
        if isinstance(save_file, str):
            fig.savefig(save_file)
        else:  # Allow passing a list
            _ = [fig.savefig(f) for f in save_file]
    plt.show()
    return
