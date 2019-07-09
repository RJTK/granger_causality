import numpy as np
import pandas as pd

import seaborn as sns
import os

from matplotlib import pyplot as plt
from levinson.levinson import compute_covariance, lev_durb
from multiprocessing.pool import ThreadPool
from threading import Lock

from pwgc.gc_methods import (pwgc_estimate_graph, get_residual_graph,
                             get_X, combine_graphs, estimate_B,
                             attach_X, compute_bic, form_Xy)
from pwgc.var_system import remove_zero_filters

from matplotlib import rc as mpl_rc
font = {"family": "normal",
        "weight": "bold",
        "size": 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=True)


# Ensure we always have the same channels and that they are in the same order.
channels = [
    "AF1", "AF2", "AF7", "AF8", "AFZ", "C1", "C2", "C3", "C4", "C5", "C6",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ", "CZ", "F1", "F2", "F3",
    "F4", "F5", "F6", "F7", "F8", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FCZ", "FP1", "FP2", "FPZ", "FT7", "FT8", "FZ", "O1", "O2", "OZ", "P1",
    "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PO1", "PO2", "PO7", "PO8",
    "POZ", "PZ", "T7", "T8", "TP7", "TP8", "X", "Y", "nd"]

# X, Y are "EOG" (Electro-oculogram) for eyes
# nd is a reference electrode

thread_lock = Lock()


def compute_all_networks():
    pool = ThreadPool(4)
    pool.map(compute_and_save_networks, os.listdir("eeg"))
    pool.close()
    pool.join()
    return


def compute_and_save_networks(subject):
    print(subject)
    if subject[3] == "a":
        pass
    elif subject[3] == "c":
        pass
    else:
        return

    Adj = estimate_subject_G_adj(subject)
    np.save("eeg/adj_estimates/" + subject + ".npy", Adj)
    return


def network_comparison_example():
    subject_control = "co2c0000337"
    subject_alcoholic = "co2a0000364"

    A_control = estimate_subject_G_adj(subject_control)
    A_alcoholic = estimate_subject_G_adj(subject_alcoholic)

    plot_adj_mat(A_control, save_file=["../figures/control_adj_mat.png",
                                       "../figures/control_adj_mat.pdf"],
                 show=True, title="Control")
    plot_adj_mat(A_alcoholic, save_file=["../figures/alcoholic_adj_mat.png",
                                         "../figures/alcoholic_adj_mat.pdf"],
                 show=True, title="Alcoholic")
    return


def estimate_subject_G_adj(subject):
    folder = "eeg/" + subject + "/"
    save_folder = "eeg/adj_matrices/" + subject + "/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_G_hat = []

    for file_name in os.listdir(folder):
        try:
            _, _, _, G_hat = estimate_eeg_graph(folder + file_name,
                                                post_estimate="lstsqr")
            all_G_hat.append(G_hat)
            A_hat = make_adj_mat([G_hat])
            np.save(save_folder + file_name + ".npy", A_hat)
        except Exception as e:
            # Here's a ghetto error logger
            err = ("Caught Exception {} while trying to estimate eeg "
                   "graph for subject {} and file {}.  We can try to "
                   "continue...".format(e, subject, file_name))
            print(err)

            thread_lock.acquire(blocking=True, timeout=-1)
            with open("eeg_errors.txt", "w+") as f:
                f.write(err + "\n")
            thread_lock.release()

    return make_adj_mat(all_G_hat)


def plot_adj_mat(G_adj, save_file=None, show=False, title=""):
    plt.imshow(G_adj, origin="upper", vmin=0, vmax=1)
    plt.xticks(ticks=np.arange(len(channels)), labels=channels,
               rotation=-90, fontsize=8)
    plt.yticks(ticks=np.arange(len(channels)), labels=channels,
               fontsize=8)

    plt.grid(False)
    plt.colorbar()
    plt.title(title)

    if save_file is not None:
        if type(save_file) is str:
            plt.savefig(save_file)
        else:
            for sf in save_file:
                plt.savefig(sf)
    if show:
        plt.show()
    return


def make_adj_mat(all_G_hat):
    n_nodes = len(all_G_hat[0])
    edges = {i: {j: 0 for j in range(n_nodes)} for i in range(n_nodes)}
    for g in all_G_hat:
        for (i, j) in g.edges:
            edges[i][j] += 1

    # G_hat = nx.DiGraph(edges)
    G_adj = np.zeros((n_nodes, n_nodes))
    for i in edges:
        for j in edges[i]:
            G_adj[i, j] = edges[i][j]
    G_adj = G_adj / len(all_G_hat)
    G_adj[np.arange(n_nodes), np.arange(n_nodes)] = 0.0
    return G_adj


def oos_error_demonstration(subject="co2a0000364"):
    all_V0 = []
    all_V = []
    all_V_bl = []
    all_G_hat = []
    folder = "eeg/" + subject + "/"
    for file_name in os.listdir(folder):
        V, V_bl, V0, G_hat = estimate_eeg_graph(folder + file_name,
                                                post_estimate="lstsqr")
        plt.plot(V, color="b", alpha=0.75)
        plt.plot(V_bl, color="r", alpha=0.75)
        plt.plot(V0, color="g", alpha=0.75)
        all_V.append(V)
        all_V_bl.append(V_bl)
        all_V0.append(V0)
        all_G_hat.append(G_hat)
    plt.show()

    V = np.vstack(all_V).ravel()
    V0 = np.vstack(all_V0).ravel()
    V_bl = np.vstack(all_V_bl).ravel()

    sns.distplot(np.log(V), label="PWGC")
    sns.distplot(np.log(V0), label="Raw Variance")
    sns.distplot(np.log(V_bl), label="Baseline (disjoint AR)")
    plt.legend()
    plt.title("Out of Sample Error (subject {})".format(subject))
    plt.xlabel("log MSE")
    plt.ylabel("Density")
    plt.savefig("../figures/eeg_oos_demonstration.png")
    plt.savefig("../figures/eeg_oos_demonstration.pdf")
    plt.show()
    return


def oos_example():
    max_T = 200
    max_lag = 5
    X = get_eeg_data("eeg/co2a0000364/co2a0000364.rd.000")
    baseline_estimates(X, max_T, max_lag, plot_example=True)
    return


def get_eeg_data(file_name):
    D = read_trial(file_name)
    X = D.loc[:, channels]\
        .to_numpy()

    X = X - np.mean(X, axis=0)
    X = X / np.std(X)
    X = X + 1e-2 * np.random.normal(size=X.shape)  # Dithering
    return X


def estimate_eeg_graph(file_name, post_estimate="lstsqr"):
    X = get_eeg_data(file_name)
    max_T = 200
    p_max = 5

    V, G_hat = full_graph_estimates(X, max_T=max_T, max_lag=p_max,
                                    N_iters=10, post_estimate=post_estimate)
    V_bl = baseline_estimates(X, max_T, p_max)
    V0 = np.var(X[max_T:], axis=0)
    return V, V_bl, V0, G_hat


def baseline_estimates(X, max_T, max_lag, plot_example=False):
    """
    Baseline estimate where we just fit AR(p) models individually
    """
    def fit_univariate_ar(x):
        r = compute_covariance(x[:, None], max_lag).ravel()
        _, _, eps = lev_durb(r)
        bic = compute_bic(eps, len(x), s=1)
        p_opt = np.argmax(bic)
        a, _, _eps = lev_durb(r[:p_opt + 1])
        assert np.isclose(_eps[-1], eps[p_opt])
        b = np.array(-a[1:])
        return b

    def oos_error(b, x_oos):
        if len(b) == 0:
            return np.var(x_oos)
        else:
            X_fit, y_oos = form_Xy(x_oos[:, None], x_oos, p=len(b))
            return np.var(y_oos - X_fit @ b)

    def baseline_var(x):
        b = fit_univariate_ar(x[:max_T])
        return oos_error(b, x[max_T - len(b):])

    V = np.array([baseline_var(X[:, i]) for i in range(X.shape[1])])

    if plot_example:
        x = X[:, 0]
        b = fit_univariate_ar(x[:max_T])
        x_oos = x[max_T - len(b):]
        X_fit, y_oos = form_Xy(x_oos[:, None], x_oos, p=len(b))
        x_hat = X_fit @ b
        T0 = 150
        t = np.arange(T0, len(x))
        plt.plot(t, x[T0:], color="b", label="signal", linewidth=2,
                 marker="o")
        plt.plot(np.arange(max_T, len(x)), x_hat,
                 color="r", linewidth=2, label="OOS 1-step-ahead Estimates",
                 marker="s")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("EEG Signal (normalized nv)")
        plt.title("Illustrative Estimates (Univariate AR)")
        plt.savefig("../figures/eeg_estimate_example.png")
        plt.savefig("../figures/eeg_estimate_example.pdf")
        plt.show()
    return V


def full_graph_estimates(X, max_T, max_lag, N_iters=4, post_estimate="lstsqr"):
    graph_estimates = []

    def _estimate(X_data):
        G_e = pwgc_estimate_graph(X[:max_T], max_lags=max_lag, method="lstsqr")
        G_r = get_residual_graph(G_e)
        X_r = get_X(G_r)
        return G_e, X_r

    sv2_r_prev = np.var(X)
    sv2_r = np.inf

    X_r = X
    for _ in range(N_iters):
        G_e, X_r = _estimate(X_r)
        graph_estimates.append(G_e)
        sv2_r = np.var(X_r)
        if sv2_r / sv2_r_prev > 0.9:
            break
        else:
            sv2_r_prev = sv2_r

    G_hat = combine_graphs(graph_estimates)
    G_hat = attach_X(G_hat, X)
    G_hat = estimate_B(G_hat, max_lag=max_lag, method=post_estimate,
                       max_T=max_T)
    G_hat = remove_zero_filters(G_hat)
    X_r = get_X(G_hat, prop="r")
    # X_original = X[max_T + max_lag:, :]
    # X_pred = X_original - X_r
    # # x_baseline = x_real[:-1]
    # V = np.var(X_original, axis=0) - np.var(X_r, axis=0)
    V = np.var(X_r, axis=0)
    return V, G_hat


def read_trial(file_name):
    eeg = pd.read_csv(file_name, comment="#", sep=" ",
                      names=["chan", "time", "nv"])
    eeg = eeg.pivot(index="time", columns="chan", values="nv")
    return eeg


if __name__== "__main__":
    # oos_example()
    # oos_error_demonstration()
    compute_all_networks()
    pass
