"""
Comparisons for PWGC and LASSO.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from plotting_helpers import COLOR1, COLOR2, plotting_model

from matplotlib import rc as mpl_rc
font = {"family": "normal",
        "weight": "bold",
        "size": 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=True)  # TODO: Fix my fonts

from pwgc.var_system import (random_gnp_dag, drive_gcg, get_errors,
                             get_estimation_errors, random_scg,
                             get_X, attach_node_prop, gcg_to_var)
from pwgc.gc_methods import (compute_pairwise_gc, estimate_B, pw_scg,
                             estimate_graph, full_filter_estimator,
                             compute_MCC_score, estimate_dense_graph,
                             alasso_fista_estimate_dense_graph)


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


def simple_lasso():
    np.random.seed(0)
    n_nodes, p_lags, p_max = 25, 2, 3

    T_iters = np.array(list(map(int, np.linspace(50, 20000, 150))))

    N_edges = (len(random_scg(n_nodes, p_lags, pole_rad=0.75).edges) -
               n_nodes)  # Due to self loops
    edge_prob = 2 * N_edges / (n_nodes * (n_nodes - 1))

    random_graph = lambda: random_gnp_dag(
        n_nodes, p_lags, pole_rad=0.75,
        edge_prob=edge_prob)

    def make_test_data(T):
        sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
        G = random_graph()
        drive_gcg(G, 500 + min(2 * T, T + 15000), sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        X = X
        return G, X, sv2_true

    errs_lasso = TrackErrors(1, T_iters, n_nodes)

    N_iter = 0
    for T_iter, T in enumerate(T_iters):
        print("T[{} / {}]".format(
            T_iter + 1, len(T_iters)),
            end="\r")
        G, X, sv2_true = make_test_data(T)
        G_hat_lasso = estimate_dense_graph(X, max_lag=p_max,
                                           max_T=T,
                                           method="alasso")
        errs_lasso.update(G, G_hat_lasso, sv2_true, N_iter, T_iter)

    D_errs_lasso, D_true_errs, D_MCC_lasso = errs_lasso.get_results()

    plot_lasso_results(D_errs_lasso, D_MCC_lasso, D_true_errs,
                       title="Test Errors against T (Random SCG graph on "
                              "$n = {}$ nodes)".format(n_nodes),
                       lasso_title="Adaptive LASSO",
                       show_results=True,
                       save_file=["../figures/lasso_only_simulation.pdf",
                                  "../figures/jpgs_pngs/lasso_only_simulation.png"])
    return


def lasso_mcc_comparison(simulation_name, showfig=False, fista=False):
    np.random.seed(0)
    n_nodes, p_lags, p_max = 50, 5, 15
    alpha = 0.05

    N_edges = (len(random_scg(n_nodes, p_lags, pole_rad=0.75).edges) -
               n_nodes)  # Due to self loops
    edge_prob_q2 = 2 * N_edges / (n_nodes * (n_nodes - 1))
    edge_prob_q4 = 4 * N_edges / (n_nodes * (n_nodes - 1))

    random_scg_graph = lambda: random_scg(
        n_nodes, p_lags, pole_rad=0.75)
    random_dag_graph_q2 = lambda: random_gnp_dag(
        n_nodes, p_lags, pole_rad=0.75,
        edge_prob=edge_prob_q2)
    random_dag_graph_q4 = lambda: random_gnp_dag(
        n_nodes, p_lags, pole_rad=0.75,
        edge_prob=edge_prob_q4)

    def make_test_data(T, graph_type="DAG"):
        sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
        if graph_type == "DAG_q2":
            G = random_dag_graph_q2()
        elif graph_type == "DAG_q4":
            G = random_dag_graph_q4()
        elif graph_type == "SCG":
            G = random_scg_graph()
        else:
            assert False

        drive_gcg(G, 500 + min(2 * T, T + 15000), sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        X = X
        return G, X, sv2_true

    def collect_data(graph_type, T_iters):
        errs_pwgc = TrackErrors(1, T_iters, n_nodes)
        errs_lasso = TrackErrors(1, T_iters, n_nodes)

        N_iter = 0
        for T_iter, T in enumerate(T_iters):
            print("T[{} / {}]".format(
                T_iter + 1, len(T_iters)),
                end="\r")

            G, X, sv2_true = make_test_data(T, graph_type)

            # When the number of samples is small we can get away with
            # using the full least squares method instead of WLD.
            fast_mode = False
            if T > 300:
                fast_mode = True

            G_hat_pwgc = estimate_graph(X[:T], G, max_lags=p_max,
                                        method="lstsqr", alpha=alpha,
                                        fast_mode=fast_mode, F_distr=True)
            if fista:
                G_hat_lasso = alasso_fista_estimate_dense_graph(
                    X, max_lag=p_max, max_T=T)
            else:
                G_hat_lasso = estimate_dense_graph(
                    X, max_lag=p_max, max_T=T, method="alasso")

            errs_pwgc.update(G, G_hat_pwgc, sv2_true, N_iter, T_iter)
            errs_lasso.update(G, G_hat_lasso, sv2_true, N_iter, T_iter)
        return (np.array(errs_pwgc.get_results()[2]["value"], dtype=float),
                np.array(errs_lasso.get_results()[2]["value"]))

    T_iters = np.array(list(map(int, np.linspace(20, 100, 200))))

    mcc_pwgc_scg, mcc_lasso_scg = collect_data("SCG", T_iters)
    mcc_pwgc_dag_q2, mcc_lasso_dag_q2 = collect_data("DAG_q2", T_iters)
    mcc_pwgc_dag_q4, mcc_lasso_dag_q4 = collect_data("DAG_q4", T_iters)

    plot_mcc_results(T_iters, mcc_pwgc_scg, mcc_lasso_scg,
                     mcc_pwgc_dag_q2, mcc_lasso_dag_q2,
                     mcc_pwgc_dag_q4, mcc_lasso_dag_q4,
                     title=("MCC Comparison on $n = {}$ nodes"
                            "".format(n_nodes)),
                     lasso_title="Adaptive LASSO", show_results=showfig,
                     save_file=["../figures/{}.pdf".format(simulation_name),
                                "../figures/jpgs_pngs/{}.png".format(simulation_name)]
                    )
    return


def lasso_comparison(simulation_name, graph_type, showfig=False, fista=False):
    np.random.seed(0)
    n_nodes, p_lags, p_max = 50, 5, 5
    alpha, N_iters = 0.05, 2

    T_iters = list(map(int, np.linspace(50, 5000, 200)))
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
        drive_gcg(G, 500 + min(2 * T, T + 15000), sv2_true, filter_attr="b(z)")
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

            G, X, sv2_true = make_test_data(T)
            G_hat_pwgc = estimate_graph(X[:T], G, max_lags=p_max,
                                        method="lstsqr", alpha=alpha,
                                        fast_mode=True, F_distr=False)

            if fista:
                G_hat_lasso = alasso_fista_estimate_dense_graph(
                    X, max_lag=p_max, max_T=T, eps=1e-3, full_path=False)
            else:
                G_hat_lasso = estimate_dense_graph(
                    X, max_lag=p_max, max_T=T, method="alasso")

            errs_pwgc.update(G, G_hat_pwgc, sv2_true, N_iter, T_iter)
            errs_lasso.update(G, G_hat_lasso, sv2_true, N_iter, T_iter)

    D_errs_pwgc, D_true_errs, D_MCC_pwgc = errs_pwgc.get_results()
    D_errs_lasso, _, D_MCC_lasso = errs_lasso.get_results()

    plot_results(
        D_MCC_pwgc, D_errs_pwgc, D_MCC_lasso, D_errs_lasso,
        D_true_errs, show_results=showfig,
        title=("Test Errors against T (Random {} graph on "
               "$n = {}$ nodes)".format(graph_type, n_nodes)),
        lasso_name="Adaptive LASSO",
        save_file=["../figures/new_{}_simulation.pdf".format(simulation_name),
                   "../figures/jpgs_pngs/"
                   "new_{}_simulation.png".format(simulation_name)])
    return


def _fit_sqrt_model(x, y, include_linear=False):
    if include_linear:
        X = np.vstack((np.ones_like(x), x, np.log(x), np.sqrt(x))).T
    else:
        X = np.vstack((np.ones_like(x), np.sqrt(x), np.sqrt(x + np.median(x)))).T

    fit = sm.OLS(endog=y, exog=X, hasconst=True)
    fit_res = fit.fit()
    y_hat = fit_res.fittedvalues
    _, y_low, y_high = wls_prediction_std(fit_res, X, alpha=0.1)
    return y_low, y_hat, y_high


def _plot_plotting_model(x, y, ax, color, marker):
    ax.scatter(x, y, color=color, marker=marker, alpha=0.75,
               linewidths=0.0)

    y_hat = plotting_model(x, y)
    ax.plot(x, y_hat, color=color, linewidth=2)
    return


def _plot_individual_results(t, var_pwgc, var_lasso, ax, ylabel="", 
                             title="", lasso_label="LASSO", legend=False):
    ax.set_title(title)
    _plot_plotting_model(t, var_pwgc, ax, COLOR1, "o")
    _plot_plotting_model(t, var_lasso, ax, COLOR2, "^")

    ax.set_xlabel("Time Points $T$")
    ax.set_ylabel(ylabel)

    if legend:
        ax.plot([], [], color=COLOR1, marker="o", label="PWGC")
        ax.plot([], [], color=COLOR2, marker="^", label=lasso_label)
        ax.legend(loc="upper left")
    ax.grid(True)
    return


# def plot_lasso_results(D_errs_lasso, D_MCC_lasso, D_true_errs,
#                        title=None, lasso_title=None, show_results=False,
#                        save_file=None):
#     true_errs = np.array(D_true_errs["value"], dtype=float)
#     t = np.array(D_true_errs["variable"], dtype=float)

#     def _D_to_np(D_MCC, D_errs):
#         mcc = np.array(D_MCC["value"], dtype=float)
#         errs = np.array(D_errs["value"], dtype=float)
#         errs = errs / true_errs  # Error relative to noise floor
#         return mcc, errs

#     mcc_lasso, errs_lasso = _D_to_np(D_MCC_lasso, D_errs_lasso)

#     fig, ax = plt.subplots(1, 1)

#     _plot_individual_results(t, mcc_lasso, errs_lasso, ax,
#                              title=lasso_title, legend=False)
#     fig.suptitle(title)

#     if save_file is not None:
#         if isinstance(save_file, str):
#             fig.savefig(save_file)
#         else:  # Allow passing a list
#             _ = [fig.savefig(f) for f in save_file]
#     if show_results:
#         plt.show()
#     return


def plot_results(D_MCC_pwgc, D_errs_pwgc,
                 D_MCC_lasso, D_errs_lasso,
                 D_true_errs,
                 save_file=None, title="",
                 lasso_name="LASSO",
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

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
    axes[0].set_ylim(0, 1.25)

    _plot_individual_results(t, mcc_pwgc, mcc_lasso, axes[0],
                             title="MCC", lasso_label=lasso_name,
                             legend=True, ylabel="MCC")
    _plot_individual_results(t, errs_pwgc, errs_lasso, axes[1],
                             title="Prediction Error",
                             ylabel="Log Prediction Error Relative to Noise Floor",
                             lasso_label=lasso_name,
                             legend=False)
    fig.suptitle(title)

    if save_file is not None:
        if isinstance(save_file, str):
            fig.savefig(save_file)
        else:  # Allow passing a list
            _ = [fig.savefig(f) for f in save_file]
    if show_results:
        plt.show()
    return


def plot_mcc_results(t, mcc_pwgc_scg, mcc_lasso_scg,
                     mcc_pwgc_dag_q2, mcc_lasso_dag_q2,
                     mcc_pwgc_dag_q4, mcc_lasso_dag_q4,
                     title=None, lasso_title=None,
                     show_results=False,
                     save_file=None):
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.suptitle(title)
    axes[0].set_ylim(0, 1)

    axes[0].set_title("SCG", loc="left")
    axes[1].set_title("DAG ($q = \\frac{2}{n}$)", loc="left")
    axes[2].set_title("DAG ($q = \\frac{4}{n}$)", loc="left")

    axes[2].set_xlabel("$T$ Samples")
    axes[0].set_ylabel("MCC")
    axes[1].set_ylabel("MCC")
    axes[2].set_ylabel("MCC")

    def plot_it(ax, mcc, c, m, l):
        ax.scatter(t, mcc, marker=m, color=c, alpha=0.75)
        ax.plot(t, plotting_model(t, mcc), color=c, linewidth=2)
        ax.plot([], [], marker=m, color=c, label=l)
        return

    plot_it(axes[0], mcc_pwgc_scg, COLOR1, "o", "PWGC Algorithm")
    plot_it(axes[0], mcc_lasso_scg, COLOR2, "^", lasso_title)
    plot_it(axes[1], mcc_pwgc_dag_q2, COLOR1, "o", "PWGC Algorithm")
    plot_it(axes[1], mcc_lasso_dag_q2, COLOR2, "^", lasso_title)
    plot_it(axes[2], mcc_pwgc_dag_q4, COLOR1, "o", "PWGC Algorithm")
    plot_it(axes[2], mcc_lasso_dag_q4, COLOR2, "^", lasso_title)

    axes[0].legend()
    plt.tight_layout()

    if save_file is not None:
        if isinstance(save_file, str):
            fig.savefig(save_file)
        else:  # Allow passing a list
            _ = [fig.savefig(f) for f in save_file]
    if show_results:
        plt.show()
    return


if __name__ == "__main__":
    # lasso_comparison("lasso_comparison_scg_pmax15_fista", "SCG", fista=True)
    # lasso_comparison("lasso_comparison_dag_pmax15_fista", "DAG", fista=True)
    # lasso_mcc_comparison(
    #     "lasso_mcc_comparison_fista", showfig=True, fista=True)
    # pass
