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
                                     GridSearchCV)
# from sklearn.learning_curves import learning_curve
from sklearn.metrics import matthews_corrcoef

from matplotlib import rc as mpl_rc
font = {"family": "normal",
        "weight": "bold",
        "size": 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=True)


def main():
    dataset = load_dataset()

    X = np.array(dataset["A"] + dataset["C"])
    y = np.append(np.ones(len(dataset["A"])),
                  np.zeros(len(dataset["C"])))
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.75)
    train_ix, valid_ix = next(ss.split(X, y))
    X_valid, y_valid = X[valid_ix], y[valid_ix]
    X, y = X[train_ix], y[train_ix]

    K_rk_apx = 20
    gk = GraphKernel(
        kernel=[{"name": "weisfeiler_lehman", "niter": 5},
                {"name": "subtree_wl"}], normalize=True,
        Nystroem=K_rk_apx,
        n_jobs=4)

    K = gk.fit_transform(X)
    param_grid = {"C": np.logspace(-5, 5, 25)}
    cv = GridSearchCV(estimator=SVC(kernel="linear"),
                      param_grid=param_grid,
                      n_jobs=1, iid=True, cv=3,
                      refit=True)
    cv.fit(K, y)

    # skf = StratifiedKFold(n_splits=3, shuffle=True)
    # C_star = None
    # score_star = -np.inf
    # for c in np.logspace(-5, 5, 25):
    #     total_score = 0
    #     for train_ix, test_ix in skf.split(X, y):
    #         K_train = gk.fit_transform(X[train_ix])
    #         K_test = gk.transform(X[test_ix])

    #         clf = SVC(C=c, kernel="percomputed", probability=False)
    #         clf.fit(K_train, y[train_ix])
    #         pred = clf.predict(K_test)
    #         total_score += matthews_corrcoef(y[test_ix], pred)

    #     if total_score > score_star:
    #         score_star = total_score
    #         C_star = clf.C

    # K = gk.fit_transform(X)
    # clf = SVC(C=C_star, kernel="percomputed", probability=True)
    # clf.fit(K, y)
    return


def load_dataset():
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
