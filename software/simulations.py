import seaborn.apionly as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import rc as mpl_rc
font = {"family" : "normal",
        "weight" : "bold",
        "size"   : 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=True)

from var_system import (random_gnp_dag, drive_gcg, get_errors,
                        get_estimation_errors, random_tree_dag,
                        get_X, attach_node_prop, gcg_to_var)
from gc_methods import (compute_pairwise_gc, estimate_B, pw_scg,
                        estimate_graph, full_filter_estimator,
                        compute_MCC_score)


def full_single_pass_experiment(simulation_name, graph_type, M_passes=1):
    n_nodes, p_lags, max_lags = 50, 5, 10
    N_iters = 50

    T_max = 11000
    # T_iters = [100, 500, 1000, 2500, 5000, 10000]
    T_iters = [100, 500, 1000, 2500, 10000]

    if graph_type == "Tree":  # TODO: This doesn't actually produce trees!
        random_graph = lambda: random_tree_dag(
            n_nodes, p_lags, pole_rad=0.75)
    elif graph_type == "DAG":
        random_graph = lambda: random_gnp_dag(
            n_nodes, p_lags, pole_rad=0.75,
            edge_prob=2. / n_nodes)
    elif graph_type == "ErdosRenyi":
        random_graph = lambda: random_gnp(
            n_nodes, p_lags, pole_rad=0.50,
            edge_prob=2. / n_nodes)
    else:
        raise NotImplementedError("graph_type '{}' is not understood"
                                  "".format(graph_type))

    b_errs = np.zeros((N_iters, len(T_iters)))
    errs = np.zeros_like(b_errs)
    errs_vec = np.zeros((N_iters, len(T_iters), n_nodes))
    true_errs = np.zeros_like(b_errs)
    true_errs_vec = np.zeros_like(errs_vec)
    rhos = np.zeros_like(b_errs)
    N_cross_edges = np.zeros_like(b_errs)
    N_hat_cross_edges = np.zeros_like(b_errs)
    N_cross_intersect_edges = np.zeros_like(b_errs)

    # TODO: There is a ton of repeated work going on all over
    for T_iter, T in enumerate(T_iters):
        for N_iter, _ in enumerate(range(N_iters)):
            print("N[{} / {}] -- T[{} / {}]\r".format(
                N_iter + 1, N_iters, T_iter + 1, len(T_iters)))

            sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
            G = random_graph()
            drive_gcg(G, T_max, sv2_true, filter_attr="b(z)")
            X = get_X(G)
            X = X - np.mean(X, axis=0)[None, :]

            # TODO: I am using T_max samples on every run!
            G_hat = estimate_graph(X, G, max_lags=10, method="lstsqr")
            draw_graph_estimates(G, G_hat)
            # G_hat = full_filter_estimator(G, M_passes=1, T_max=T)
            # X = get_X(G, prop="x")

            # # We pass in G only to attach "x" to G_hat
            # G_hat = estimate_graph(X[:T], G)

            b_err = np.mean(list(map(np.var, get_estimation_errors(G, G_hat))))
            err = np.sum(get_errors(G_hat))
            err_vec = np.sort(get_errors(G_hat))
            true_err = np.sum(sv2_true)
            true_err_vec = np.sort(sv2_true)

            rhos[N_iter, T_iter] = gcg_to_var(
                G_hat, "b_hat(z)", assert_stable=False).get_rho()

            b_errs[N_iter, T_iter] = b_err
            errs[N_iter, T_iter] = err
            errs_vec[N_iter, T_iter] = err_vec
            true_errs[N_iter, T_iter] = true_err
            true_errs_vec[N_iter, T_iter] = true_err_vec
            N_cross_edges[N_iter, T_iter] = len(
                set((i, j) for i, j in G.edges if i != j))
            N_hat_cross_edges[N_iter, T_iter] = len(set(
                (i, j) for i, j in G_hat.edges if i != j))
            N_cross_intersect_edges[N_iter, T_iter] = len(
                (set(G.edges) & set(G_hat.edges)) - set((i, i) for i in G.nodes))

    MCC = compute_MCC_score(N_cross_edges, N_hat_cross_edges,
                            N_cross_intersect_edges, n_nodes)

    def to_pd(X, var_name=""):
        D = pd.DataFrame(X, columns=["{}".format(T) for T in T_iters]).melt()
        return D

    D_errs = to_pd(np.log(errs))
    D_true_errs = to_pd(np.log(true_errs))
    D_MCC = to_pd(MCC)

    for T_iter, c in zip(range(len(T_iters)), ["b", "r", "g", "y", "m", "lightblue"]):
        for N_iter, _ in enumerate(range(N_iters)):
            plt.plot(np.sort(errs_vec[:, T_iter, :], axis=0),
                     color=c, linewidth=0.75, alpha=0.75)
    plt.show()

    plot_results(
        D_MCC, D_errs, D_true_errs, T_iters,
        title=("Test Errors against T (Random Tree graph on "
               "$n = {}$ nodes)".format(n_nodes)))

    # plot_results(
    #     D_MCC, D_errs, D_true_errs, T_iters,
    #     title=("Test Errors against T (Random {} graph on "
    #            "$n = {}$ nodes)".format(graph_type, n_nodes)),
    #     save_file=["../figures/{}_simulation.pdf".format(simulation_name),
    #                "../figures/jpgs_pngs/{}_simulation.png".format(simulation_name)])
    return


def plot_results(D_MCC, D_errs, D_true_errs, T_iters,
                 save_file=None, title=""):
    fig, ax_mcc = plt.subplots(1, 1)
    ax_pred = ax_mcc.twinx()
    order = ["{}".format(T) for T in T_iters]

    # MCC
    sns.catplot(x="variable", y="value", data=D_MCC,
                kind="point", ax=ax_mcc, color="b",
                order=order, ci=None)
    sns.catplot(x="variable", y="value", data=D_MCC,
                kind="strip", ax=ax_mcc, color="b",
                order=order)

    # True Errors
    sns.catplot(x="variable", y="value", data=D_true_errs,
                kind="point", ax=ax_pred, color="k",
                order=order, ci=None)

    # Acheived Errors
    sns.catplot(x="variable", y="value", data=D_errs,
                kind="point", ax=ax_pred, color="r",
                order=order, ci=None)
    sns.catplot(x="variable", y="value", data=D_errs,
                kind="strip", ax=ax_pred, color="r",
                order=order)

    ax_mcc.set_xlabel("Time Points $T$")
    ax_mcc.set_ylabel("Graph Recovery MCC Score")
    ax_pred.set_ylabel("Log Prediction Error")
    ax_pred.plot([], [], color="b", label="MCC")
    ax_pred.plot([], [], color="r", label="Prediction Error")
    ax_pred.plot([], [], color="k", label="Noise Floor")
    ax_pred.legend()
    ax_pred.grid(False)

    ax_mcc.set_title(title)

    if save_file is not None:
        if isinstance(save_file, str):
            fig.savefig(save_file)
        else:  # Allow passing a list
            [fig.savefig(f) for f in save_file]
    plt.show()
    return
