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
from sklearn.linear_model import LogisticRegressionCV

from matplotlib import rc as mpl_rc
font = {"family": "normal",
        "weight": "bold",
        "size": 22}

mpl_rc("font", **font)
mpl_rc("text", usetex=False)


def main_Xy2():
    X, y, z = load_subject_Xyz()
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 2
    for i in range(10):
        xform = Pipeline(
            steps=[("pca", KernelPCA(n_components=5,
                                     kernel="rbf",
                                     gamma=0.75,
                                     eigen_solver="arpack",
                                     random_state=i,
                                     )),
                   ("power_transform", PowerTransformer(method="yeo-johnson",
                                                        standardize=True))])
        X_pca = xform.fit_transform(X)
        D_pca = pd.DataFrame(X_pca)
        D_pca["label"] = ["Alcoholic" if yi == 1 else "Control" for yi in y]
        pp = sns.pairplot(D_pca.iloc[::2, ], hue="label",
                          vars=[c for c in D_pca.columns if c != "label"],
                          plot_kws={"alpha": 0.25})
        plt.title(i)
        plt.show()

    # ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    # train_ix, valid_ix = next(ss.split(X, y))
    # X_valid, y_valid = X[valid_ix], y[valid_ix]
    # X_train, y_train = X[train_ix], y[train_ix]

    # K = Nystroem(kernel="sigmoid", gamma=0.75, n_components=200).fit_transform(K)
    # pt = PowerTransformer(method="yeo-johnson",
    #                       standardize=True)
    # K = polynomial_kernel(X_train, X_train, degree=3)

    subjects = np.unique(y)
    s0 = subjects[9]
    s1 = subjects[-1]

    sel = np.logical_or(y == s0, y == s1)
    _y = y[sel]
    _X = X[sel]
    qt = QuantileTransformer(output_distribution="normal")
    K = Nystroem(kernel="poly", gamma=0.75, degree=4, coef0=1.0, n_components=500,
                 random_state=0)\
                 .fit_transform(_X)
    K = qt.fit_transform(K)

    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(K, _y)
    X_pls = pls.x_scores_
    plt.scatter(X_pls[:, 0], X_pls[:, 1], c=_y)
    plt.show()

    K = Nystroem(kernel="poly", gamma=0.75, degree=4, coef0=1.0, n_components=50,
                 random_state=0)\
                 .fit_transform(X_train)
    qt = QuantileTransformer(output_distribution="normal")
    K = qt.fit_transform(K)
    clf = LogisticRegressionCV(Cs=20,
                               multi_class="multinomial",
                               cv=3, dual=False, solver="saga",
                               refit=True)
    clf.fit(K, y_train)

    clf = SVC(kernel="poly", degree=4)
    param_grid = {"gamma": np.logspace(-2, 2, 25)}

    cv = RandomizedSearchCV(estimator=clf,
                            param_distributions=param_grid,
                            n_iter=1, n_jobs=1, cv=2)
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_ix, valid_ix = next(ss.split(X, y))
    X_valid, y_valid = X[valid_ix], y[valid_ix]
    X_train, y_train = X[train_ix], y[train_ix]
    cv.fit(X_train[::5], y_train[::5])

    plt.scatter(pls.x_scores_[:, 0], pls.x_scores_[:, 1],
                alpha=0.5, c=y)
    plt.show()

    param_grid = {"n_estimators": [650],
                  "learning_rate": [0.055],
                  "base_estimator": [DecisionTreeClassifier(max_depth=1)]}
    cv = GridSearchCV(estimator=AdaBoostClassifier(),
                      param_grid=param_grid,
                      n_jobs=3, iid=True, cv=3,
                      refit=True)
    cv.fit(X_train, y_train)

    y_hat_valid = cv.predict(X_valid)
    # y_hat_valid = cv.fit(X_pca, y).predict(X_pca_valid)
    # clf = clf.fit(X_train[::5], y_train[::5])
    # y_hat_valid = clf.predict(X_valid)
    acc_valid = np.mean(y_valid == y_hat_valid)
    print(acc_valid)
    return


def main_Xy():
    X, y = load_Xy_dataset()

    xform = Pipeline(
        steps=[("pca", KernelPCA(n_components=3,
                                 kernel="rbf",
                                 gamma=0.75,
                                 eigen_solver="arpack",
                                 random_state=0,
                                 )),
               ("power_transform", PowerTransformer(method="yeo-johnson",
                                                    standardize=True))])
    X_pca = xform.fit_transform(X)
    D_pca = pd.DataFrame(X_pca)
    D_pca["label"] = ["Alcoholic" if yi == 1 else "Control" for yi in y]
    D_pca = D_pca[::5]
    pp = sns.pairplot(D_pca, hue="label",
                      vars=[c for c in D_pca.columns if c != "label"],
                      plot_kws={"alpha": 0.25})
    plt.show()

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_ix, valid_ix = next(ss.split(X, y))
    X_valid, y_valid = X[valid_ix], y[valid_ix]
    X_train, y_train = X[train_ix], y[train_ix]

    # pca = KernelPCA(n_components=6, kernel="rbf", gamma=0.75)
    # X_pca = pca.fit_transform(X_train)
    # # X_pca_valid = pca.transform()
    # std = np.std(X_pca, axis=0)
    # X_pca = X_pca / std
    # X_pca = X_pca - np.mean(X_pca, axis=0)
    # # x_pca_valid = X_pca_valid / std

    # D_pca = pd.DataFrame(X_pca)
    # D_pca["label"] = ["Alcoholic" if yi == 1 else "Control" for yi in y]
    # pp = sns.pairplot(D_pca, hue="label",
    #                   vars=[c for c in D_pca.columns if c != "label"],
    #                   plot_kws={"alpha": 0.25})
    # plt.show()

    # param_grid = {# "svm__C": np.linspace(1.0, 5.0, 250),
    #               "svm__gamma": np.linspace(0.5, 1.5, 250),
    #               "svm__C": np.linspace(1.0, 5.0, 250)}
    #               # "svm__gamma": np.linspace(0.5, 1.5, 250),
    #               # "pca__gamma": np.linspace(0.1, 1.5, 15)}
    # cv = RandomizedSearchCV(estimator=clf,
    #                         param_distributions=param_grid,
    #                         n_jobs=6, iid=True, cv=3,
    #                         refit=True, n_iter=15)
    # cv = GridSearchCV(estimator=clf,
    #                   param_grid=param_grid,
    #                   n_jobs=3, iid=True, cv=3, refit=True
    #                   )
    # cv.fit(X_train, y_train)

    # acc = []
    # ssf = StratifiedShuffleSplit(n_splits=125, test_size=0.25, train_size=0.25)
    # for train_ix, test_ix in ssf.split(X_pca, y):
    #     X_train, y_train = X_pca[train_ix], y[train_ix]
    #     cv.fit(X_train, y_train)
    #     y_hat = cv.best_estimator_.predict(X_pca[test_ix])
    #     acc.append(np.mean(y_hat == y[test_ix]))

    # train_sizes = np.linspace(0.01, 0.5, 50)
    # fig, ax = plot_learning_curve(
    #     cv.best_estimator_,
    #     "Learning Curves SVC (best estimator from 5-fold CV)",
    #     X_pca, y, cv=10, train_sizes=train_sizes)
    # ax.legend()
    # # fig.savefig("../figures/svc_learning_curve.png")
    # # fig.savefig("../figures/svc_learning_curve.pdf")
    # plt.show()

    param_grid = {"n_estimators": [650],
                  "learning_rate": [0.055],
                  "base_estimator": [DecisionTreeClassifier(max_depth=1)]}
    cv = GridSearchCV(estimator=AdaBoostClassifier(),
                      param_grid=param_grid,
                      n_jobs=3, iid=True, cv=3,
                      refit=True)

    cv.fit(X_train, y_train)

    y_hat_valid = cv.predict(X_valid)
    # y_hat_valid = cv.fit(X_pca, y).predict(X_pca_valid)
    # clf = clf.fit(X_train[::5], y_train[::5])
    # y_hat_valid = clf.predict(X_valid)
    acc_valid = np.mean(y_valid == y_hat_valid)
    print(acc_valid)
    return


def main():
    # dataset = load_dataset(avg_subjects=False)
    # dataset = load_s2_dataset()

    X = np.array(dataset["A"] + dataset["C"])
    y = np.append(np.ones(len(dataset["A"])),
                  np.zeros(len(dataset["C"])))
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    train_ix, valid_ix = next(ss.split(X, y))
    X_valid, y_valid = X[valid_ix], y[valid_ix]
    X, y = X[train_ix], y[train_ix]

    K_rk_apx = 350
    gk = GraphKernel(
        kernel=[{"name": "weisfeiler_lehman", "niter": 15},
                {"name": "subtree_wl"},
                # {"name": "propagation"}
                # {"name": "shortest_path", "with_labels": True}
                ],
        normalize=True,
        Nystroem=K_rk_apx,
        n_jobs=2)
    # gk2 = GraphKernel(
    #     kernel=[{"name": "random_walk",
    #              "lamda": 0.5, "method_type": "baseline",
    #              "with_labels": True
    #              },
    #             # {"name": "propagation"}
    #             # {"name": "shortest_path", "with_labels": True}
    #             ],
    #     normalize=True,
    #     Nystroem=K_rk_apx,
    #     n_jobs=2)

    K = gk.fit_transform(X)
    # K2 = gk2.fit_transform(X)

    # iso = Isomap(n_components=2, n_neighbors=5)
    # K_iso = iso.fit_transform(K)
    # plt.scatter(K_iso[:, 0], K_iso[:, 1], c=y)
    # plt.show()

    # tsne = TSNE(n_components=2)
    # K_tsne = tsne.fit_transform(K)
    # plt.scatter(K_tsne[:, 0], K_tsne[:, 1], c=y)
    # plt.show()

    param_grid = {"C": np.logspace(-1, 1, 25)}
                  # "gamma": np.logspace(-6, 1, 35)}
    cv = GridSearchCV(estimator=SVC(kernel="linear"),
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


def test():
    N = 4000
    p = 0.4
    N1 = int(p * N)
    N2 = int((1 - p) * N)

    n = 8
    s = 1.75
    L = np.random.normal(size=(n, 2 * n))
    L = L @ L.T
    X1 = np.random.normal(size=(N1, n)) @ L
    X2 = s * np.random.normal(size=(N2, n)) @ L
    X = np.vstack((X1, X2))
    y = np.append(np.zeros(N1), np.ones(N2))

    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    # xform = Pipeline(
    #     steps=[("pca", KernelPCA(n_components=4,
    #                              kernel="rbf", gamma=0.75)),
    #            ("power_transform", PowerTransformer(method="yeo-johnson",
    #                                                 standardize=True))])
    # X_xform = xform.fit_transform(X)
    # plt.scatter(X_xform[:, 0], X_xform[:, 1], c=y, alpha=0.5)
    # plt.show()

    # clf = Pipeline(
    #     steps=[("pca", KernelPCA(n_components=6,
    #                              kernel="rbf")),
    #            ("power_transform", PowerTransformer(method="yeo-johnson",
    #                                                 standardize=True)),
    #            ("qda", QuadraticDiscriminantAnalysis())
    #            # ("svm", SVC(kernel="rbf"))
    #     ])

    clf = Pipeline(
        steps=[("power_transform", PowerTransformer(method="yeo-johnson",
                                                    standardize=True)),
               ("dtc", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                                          n_estimators=250))
        ])

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_ix, valid_ix = next(ss.split(X, y))
    X_valid, y_valid = X[valid_ix], y[valid_ix]
    X_train, y_train = X[train_ix], y[train_ix]

    # param_grid = {"svm__C": np.linspace(1.0, 5.0, 250),
    #               "svm__gamma": np.linspace(0.5, 1.5, 250),
    #               "pca__gamma": np.linspace(0.1, 1.5, 250)}
    # param_grid = {"pca__gamma": np.linspace(0.65, 1.15, 25)}
    #               # "knn__p": np.linspace(1.0, 5, 25)}
    # cv = RandomizedSearchCV(estimator=clf,
    #                         param_distributions=param_grid,
    #                         n_jobs=1, iid=True, cv=3,
    #                         refit=True, n_iter=10)
    # cv = cv.fit(X_train, y_train)

    clf.fit(X_train, y_train)
    y_hat_valid = clf.predict(X_valid)

    # y_hat_valid = cv.predict(X_valid)
    acc_valid = np.mean(y_hat_valid == y_valid)
    print(acc_valid)
    return

# GC = np.zeros((64, 64))
# for g in dataset["C"]:
#     GC = GC + g[0].toarray()

# GC = GC / len(dataset["C"])

# GA = np.zeros((64, 64))
# for g in dataset["A"]:
#     GA = GA + g[0].toarray()

# GA = GA / len(dataset["A"])

# fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
# axes[0].imshow(GC)
# axes[1].imshow(GA)
# plt.show()

# plt.imshow(GA - GC)
# plt.colorbar()
# plt.show()
