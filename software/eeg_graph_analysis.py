import numpy as np

import seaborn as sns
import networkx as nx
import os
import grakel
import pandas as pd

from grakel import GraphKernel  # Graph kernels
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix as sparse_matrix
from eeg_analysis import channels

from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedShuffleSplit, StratifiedKFold,
                                     GridSearchCV, learning_curve,
                                     RandomizedSearchCV)
# from sklearn.learning_curves import learning_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import Isomap, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import NMF
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import chi2_kernel, polynomial_kernel, rbf_kernel
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap, TSNE, MDS, SpectralEmbedding
from sklearn.metrics import f1_score, matthews_corrcoef

from matplotlib import rc as mpl_rc
font = {"family": "normal",
        "weight": "bold",
        "size": 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=True)


def logistic_regression_clf():
    """
    Compute and quantify the quality of a multinomial logistic regression
    model between all subjects.
    """
    X, y, z = load_subject_Xyz()
    le = LabelEncoder()
    y = le.fit_transform(y)

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_ix, valid_ix = next(ss.split(X, y))
    X_valid, y_valid = X[valid_ix], y[valid_ix]
    X_train, y_train = X[train_ix], y[train_ix]

    clf = Pipeline(
        steps=[("kern", Nystroem(kernel="poly")),
               ("qt", QuantileTransformer(output_distribution="normal")),
               ("logistic", LogisticRegression(dual=False, solver="lbfgs",
                                               class_weight="balanced",
                                               multi_class="multinomial",
                                               tol=1e-3,
                                               max_iter=500, warm_start=True))])
    param_grid = {"kern__gamma": np.linspace(0.2, 1.5, 20),
                  "kern__degree": [3, 4, 5],
                  "kern__coef0": [0.5, 1.0, 1.5],
                  "kern__n_components": [30, 40, 50, 60],
                  "logistic__C": np.logspace(-1, 1, 20),
                  "logistic__penalty": ["l2"]}

    cv = RandomizedSearchCV(estimator=clf, param_distributions=param_grid,
                            n_iter=75, n_jobs=3, cv=3, refit=True)
    cv.fit(X_train, y_train)

    y_hat_valid = cv.predict(X_valid)
    n_subjects = len(np.unique(y_valid))
    C = np.zeros((n_subjects, n_subjects))
    for label, pred in zip(y_valid, y_hat_valid):
        C[label, pred] += 1
    C = C / np.sum(C, axis=1)[:, None]
    plt.imshow(C)
    plt.colorbar()
    plt.title("Multiclass Confusion Matrix ($MCC = {:+.3f}$)"
              "".format(matthews_corrcoef(y_hat_valid, y_valid)))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("../figures/logistic_regression_confusion_matrix.pdf",
                bbox_inches="tight", pad_inches=0)
    plt.savefig("../figures/logistic_regression_confusion_matrix.png",
                bbox_inches="tight", pad_inches=0)
    plt.show()
    return


def logistic_regression_subject_pair_visualization(i=59, j=60):
    """
    Visualize the separation between GCGs of subjects i, j.  This
    visualization uses the whole dataset and likely is a slight
    overestimation of the quality of a discriminator between i, j.
    """
    X, y, z = load_subject_Xyz()
    le = LabelEncoder()
    y = le.fit_transform(y)

    subjects = np.unique(y)
    s0 = subjects[i]
    s1 = subjects[j]

    sel = np.logical_or(y == s0, y == s1)
    _y = y[sel]
    _X = X[sel]

    qt = QuantileTransformer(output_distribution="normal")
    kern = Nystroem(kernel="poly", gamma=0.55, degree=4, coef0=1.0, n_components=40,
                    random_state=0)
    K = kern.fit_transform(_X)
    K = qt.fit_transform(K)

    # pca = PCA(n_components=2)
    # X_proj = pca.fit_transform(K)

    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(K, _y)
    X_proj = pls.x_scores_

    clf = SVC(kernel="rbf")
    param_grid = {"C": np.logspace(-2, 2, 20),
                  "gamma": np.logspace(-2, 2, 20)}

    cv = GridSearchCV(estimator=clf, param_grid=param_grid,
                      n_jobs=3, cv=3, refit=True)
    cv.fit(X_proj, _y)

    clf = cv.best_estimator_
    clf.probability=True
    clf.fit(X_proj, _y)

    h = .01
    x_min, x_max = X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1
    y_min, y_max = X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
    Z = Z.reshape(xx.shape)

    sel = _y == s0
    plt.scatter(X_proj[sel, 0], X_proj[sel, 1], c="r", marker="o",
                label=le.inverse_transform([s0])[0])
    plt.scatter(X_proj[~sel, 0], X_proj[~sel, 1], c="b", marker="^",
                label=le.inverse_transform([s1])[0])
    plt.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.5,
                 vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Separating Subjets by EEG Granger-Causality Graph")
    plt.ylabel("PLS Projection $y$")
    plt.xlabel("PLS Projection $x$")
    plt.legend()
    plt.savefig("../figures/logistic_regression_separation.png",
                bbox_inches="tight", pad_inches=0)
    plt.savefig("../figures/logistic_regression_separation.pdf",
                bbox_inches="tight", pad_inches=0)
    plt.show()

    return


def logistic_regression_subject_pair(i=59, j=60, clf=None):
    """
    Compute and return the MCC distinguishing between
    subjects i and j.  We also return the classifier we fit.
    """
    X, y, z = load_subject_Xyz()
    le = LabelEncoder()
    y = le.fit_transform(y)

    subjects = np.unique(y)
    s0 = subjects[i]
    s1 = subjects[j]

    sel = np.logical_or(y == s0, y == s1)
    _y = y[sel]
    _X = X[sel]

    if clf is None:
        clf = Pipeline(
            steps=[("kern", Nystroem(kernel="poly")),
                   ("qt", QuantileTransformer(output_distribution="normal")),
                   ("logistic", LogisticRegression(dual=False, solver="lbfgs",
                                                   class_weight="balanced",
                                                   multi_class="multinomial",
                                                   tol=1e-3,
                                                   max_iter=500, warm_start=True))])
        param_grid = {"kern__gamma": np.linspace(0.2, 1.5, 20),
                      "kern__degree": [3, 4, 5],
                      "kern__coef0": [0.5, 1.0, 1.5],
                      "kern__n_components": [30, 40, 50, 60],
                      "logistic__C": np.logspace(-1, 1, 20),
                      "logistic__penalty": ["l2"]}

        cv = RandomizedSearchCV(estimator=clf, param_distributions=param_grid,
                                n_iter=75, n_jobs=3, cv=3, refit=True)
        cv.fit(X_train, y_train)
        final_clf = cv.best_estimator_
    else:
        final_clf = clf.fit(X_train, y_train)

    y_hat_valid = final_clf.predict(X_valid)
    return matthews_corrcoef(y_valid, y_hat_valid), final_clf


def load_Xy_dataset():
    dataset = load_dataset()
    X = np.vstack([dataset["A"][i][0].toarray().ravel()
                   for i in range(len(dataset["A"]))])
    Y = np.vstack([dataset["C"][i][0].toarray().ravel()
                   for i in range(len(dataset["C"]))])
    y = np.append(np.ones(len(X)), np.zeros(len(Y)))
    X = np.vstack((X, Y))
    return X, y


def load_dataset(avg_subjects=False):
    dataset = {"A": [], "C": []}
    # labels = {chan: chan for chan in channels}
    vertex_labels = {i: chan for i, chan in enumerate(channels)}

    adj_mat_folder = "eeg/adj_matrices/"
    for folder in os.listdir(adj_mat_folder):
        if folder[3] == "a":
            label = "A"
        elif folder[3] == "c":
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


def load_subject_Xyz():
    dataset = {}
    # labels = {chan: chan for chan in channels}
    vertex_labels = {i: chan for i, chan in enumerate(channels)}

    X = []
    y = []
    z = []

    adj_mat_folder = "eeg/adj_matrices/"
    for folder in os.listdir(adj_mat_folder):
        if folder[3] == "a":
            label = "A"
        elif folder[3] == "c":
            label = "C"
        else:
            continue

        _dataset = _load_all_graphs(folder, adj_mat_folder, {label: []},
                                    label, vertex_labels)

        dataset[folder] = []
        for g in [_dataset[label][i][0] for i in range(len(_dataset[label]))]:
            g = g.toarray()

            X.append(g.ravel())
            y.append(folder)
            z.append(label)
    X = np.vstack(X)
    y = np.array(y)
    z = np.array(z)
    return X, y, z


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


def load_s2_dataset():
    """
    Dataset based on whether S2 was "match" or "nomatch"
    """
    dataset = {"A": [], "C": []}
    vertex_labels = {i: chan for i, chan in enumerate(channels)}

    adj_mat_folder = "eeg/adj_matrices/"
    for folder in os.listdir(adj_mat_folder):
        for adj_file in os.listdir(adj_mat_folder + folder):
            if adj_file[-3:] != "npy":
                continue
            file_name = os.path.join(adj_mat_folder, folder, adj_file)

            trial_type = get_trial_type(folder, adj_file[:-4])
            if trial_type == "S1":
                continue
            elif trial_type == "match":
                label = "A"
            elif trial_type == "nomatch":
                label = "C"
            else:
                raise AssertionError("???")

            adj = np.load(file_name)
            adj = sparse_matrix(adj)
            dataset[label].append([adj, vertex_labels])
    return dataset


def get_trial_type(folder, file_name):
    data_folder = "eeg/"
    file_name = os.path.join(data_folder, folder, file_name)
    with open(file_name) as f:
        [f.readline() for _ in range(3)]
        l = f.readline()
        if "nomatch" in l:
            return "nomatch"
        elif "match" in l:
            return "match"
        else:
            return "S1"
    return


def adj_mat_to_graph(adj):
    g = {chan: [] for chan in channels}
    for i, chan_from in enumerate(channels):
        for j, chan_to in enumerate(channels):
            if adj[i, j]:
                g[chan_from].append(chan_to)
    return g


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 15)):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    ax.hlines(y=np.mean(y), xmin=train_sizes[0], xmax=train_sizes[-1],
              color="k", linestyle="--", label="Population Prevalence",
              linewidth=2)
    return fig, ax


if __name__ == "__main__":
    logistic_regression_clf()

    i, j = 0, 1
    logistic_regression_subject_pair_visualization(i, j)
    logistic_regression_subject_pair(i, j)
    pass

