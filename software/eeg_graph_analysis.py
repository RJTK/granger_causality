import numpy as np

import seaborn as sns
import networkx as nx
import os
import grakel

from grakel import GraphKernel  # Graph kernels
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix as sparse_matrix
from eeg_analysis import channels

from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedShuffleSplit, StratifiedKFold,
                                     GridSearchCV, learning_curve)
# from sklearn.learning_curves import learning_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import Isomap, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from matplotlib import rc as mpl_rc
font = {"family": "normal",
        "weight": "bold",
        "size": 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=False)


def main():
    dataset = load_dataset(avg_subjects=True)

    X = np.array(dataset["A"] + dataset["C"])
    y = np.append(np.ones(len(dataset["A"])),
                  np.zeros(len(dataset["C"])))
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    train_ix, valid_ix = next(ss.split(X, y))
    X_valid, y_valid = X[valid_ix], y[valid_ix]
    X, y = X[train_ix], y[train_ix]

    K_rk_apx = False
    gk = GraphKernel(
        kernel=[{"name": "weisfeiler_lehman", "niter": 5},
                # {"name": "subtree_wl"},
                {"name": "propagation"}
                # {"name": "shortest_path", "with_labels": True}
                ],
        normalize=True,
        Nystroem=K_rk_apx,
        n_jobs=4)

    K = gk.fit_transform(X)

    # lda = LinearDiscriminantAnalysis(n_components=3)
    # K_lda = lda.fit_transform(K, y)

    iso = Isomap(n_components=2, n_neighbors=5)
    K_iso = iso.fit_transform(K)
    plt.scatter(K_iso[:, 0], K_iso[:, 1], c=y)
    plt.show()

    tsne = TSNE(n_components=2)
    K_tsne = tsne.fit_transform(K)
    plt.scatter(K_tsne[:, 0], K_tsne[:, 1], c=y)
    plt.show()

    param_grid = {"C": np.logspace(-6, 1, 35),
                  "gamma": np.logspace(-6, 1, 35)}
    cv = GridSearchCV(estimator=SVC(kernel="rbf"),
                      param_grid=param_grid,
                      n_jobs=1, iid=True, cv=3,
                      refit=True)
    cv.fit(K, y)

    plot_learning_curve(cv.best_estimator_, "Learning Curves SVC",
                        K, y, cv=10)
    plt.show()
    K_valid = gk.transform(X_valid)
    y_hat_valid = cv.predict(K_valid)
    acc_valid = np.sum((y_valid == y_hat_valid) / len(y_valid))
    print(acc_valid)
    return


def load_dataset(avg_subjects=False):
    dataset = {"A": [], "C": []}
    # labels = {chan: chan for chan in channels}
    vertex_labels = {i: chan for i, chan in enumerate(channels)}

    adj_mat_folder = "eeg/adj_matrices/"
    for folder in os.listdir(adj_mat_folder):
        if folder[:4] == "co2a":
            label = "A"
        elif folder[:4] == "co2c":
            label = "C"
        else:
            continue

        if avg_subjects:
            dataset = _load_avg_graphs(folder, adj_mat_folder, dataset,
                                       label, vertex_labels)
        else:
            dataset = _load_all_graphs(folder, adj_mat_folder, dataset,
                                       label, vertex_labels)
    return dataset


def _load_avg_graphs(folder, adj_mat_folder, dataset,
                     label, vertex_labels):
    _dataset = {label: []}
    _dataset = _load_all_graphs(folder, adj_mat_folder, _dataset,
                                label, vertex_labels)
    n_nodes = len(vertex_labels)
    g = np.zeros((n_nodes, n_nodes))

    for adj, _ in _dataset[label]:
        adj = np.array(adj.todense())
        g = g + adj
    g /= n_nodes
    dataset[label].append([g, vertex_labels])
    return dataset


def _load_all_graphs(folder, adj_mat_folder, dataset,
                     label, vertex_labels):
    for adj_file in os.listdir(adj_mat_folder + folder):
        if adj_file[-3:] != "npy":
            continue
        try:
            file_name = os.path.join(adj_mat_folder, folder, adj_file)
            adj = np.load(file_name)
            adj = sparse_matrix(adj)
            # g = adj_mat_to_graph(adj)
            dataset[label].append([adj, vertex_labels])
        except Exception as e:
            print("Caught Exception {}, continuing...".format(e))
    return dataset


def adj_mat_to_graph(adj):
    g = {chan: [] for chan in channels}
    for i, chan_from in enumerate(channels):
        for j, chan_to in enumerate(channels):
            if adj[i, j]:
                g[chan_from].append(chan_to)
    return g


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
