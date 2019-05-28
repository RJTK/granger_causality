import networkx as nx
import numpy as np
from scipy import linalg

from sklearn.linear_model import LinearRegression, LassoLarsIC
from multiprocessing import Pool

from itertools import product
from math import ceil

def estimate_b_lstsqr(X, y):
    """
    Simple direct estimate of a linear filter y_hat = Xb.

    y: np.array (T)
    X: np.array (T x s)
    """
    lr = LinearRegression(fit_intercept=False, normalize=False,
                          copy_X=False, n_jobs=1)
    lr.fit(X, y)
    return lr.coef_


def estimate_b_lasso(X, y, criterion="bic"):
    """
    Estimate y_hat = Xb via LASSO and using BIC to choose regularizers.

    y: np.array (T)
    X: np.array (T x s)
    """
    lasso = LassoLarsIC(criterion=criterion, fit_intercept=False,
                        normalize=False, precompute=True)
    lr = LinearRegression(fit_intercept=False, normalize=False,
                          copy_X=False)

    # NOTE: This takes 5 - 10 ms for (5000, 125) matrix X
    lasso.fit(X, y)
    w = lasso.coef_
    if np.all(w == 0):
        # All zero support
        return w

    # Use lasso only to select the support
    X_lasso = X[:, [i for i, wi in enumerate(w) if wi != 0]]
    w_lr = lr.fit(X_lasso, y).coef_
    w[[i for i, wi in enumerate(w) if wi != 0]] = w_lr
    return w


def form_Xy(X_raw, y_raw, p=10):
    """
    Builds appropriate (X, y) matrices for autoregressive prediction.

    X_raw: np.array (T, s)
    y_raw: np.array (T)
    p: The number of lagged values to include in AR predictions.

    We organize X by collecting together lags of the same variable
    side by side -- this avoid the issues with "weaving" of
    coefficients.

    returns: X, y
    y: np.array (T - p, 1)
    X: np.array (T - p, p * s)
    """
    T, s = X_raw.shape

    # The target variable
    y = y_raw[p:]

    # Stack together lags of the same variables
    X = np.hstack(
        [np.hstack([X_raw[p - tau: -tau, j][:, None] for tau in range(1, p + 1)])
         for j in range(s)])

    assert X.shape == (T - p, s * p), "Wrong _X shape!"
    assert len(y) == T - p, "wrong y shape!"
    return X, y


def estimate_B(G, max_lag=10, copy_G=False,
               max_T=None, ic="bic"):
    """
    Estimate x(t) = B(z)x(t) respecting the topology of G.

    This method mutates G unless copy_G is True
      1. G.graph["max_lag"] = max_lag
      2. G[j][i]["b_hat(z)"] = np.array (max_lag) filter i --b(z)--> j
      3. G.nodes[i]["sv^2_hat"] = np.var(y - X @ b); the error

    G: nx.DiGraph -- should always have self-loops

    G should have data for each node attached as an attribute i.e.
    G[i]["x"]

    We use only up to max_T data points from each node to compute the
    estimates.  This allows showing the improvement with increasing T.
    """
    if copy_G:
        G = G.copy()

    G.graph["max_lag"] = max_lag

    # TODO: I could store covariance estimates on the edges
    # TODO: and pass in precomputed gram matrices to the estimators

    # TODO: There is an insane amount of repeated work here

    for i in G.nodes:
        # Form an "un-lagged" data matrix
        a_i = list(G.predecessors(i))  # Inputs to i
        y_raw = G.nodes[i]["x"]

        if len(a_i) == 0:
            G.nodes[i]["sv^2_hat"] = np.var(y_raw)
            continue
        else:
            X_raw = np.hstack([G.nodes[j]["x"][:, None] for j in a_i])

        # Form linear regression matrices
        X, y = form_Xy(X_raw, y_raw, p=max_lag)

        # Estimate
        b = estimate_b_lasso(X[:max_T], y[:max_T], criterion=ic)

        # Add the filters as properties to G
        for grouping, j in enumerate(a_i):  # Edge j --b(z)--> i
            G[j][i]["b_hat(z)"] = b[grouping * max_lag: (grouping + 1) * max_lag]
        G.nodes[i]["sv^2_hat"] = np.var(y - X @ b)
    return G


# NOTE: forming the Xy matrices consumes only microseconds
def univariate_AR_error(x, max_lag=10, criterion="bic",
                        method="lasso"):
    X, y = form_Xy(X_raw=x[:, None], y_raw=x, p=max_lag)
    return _compute_AR_error(X, y, criterion, method)


def bivariate_AR_error(y, x, max_lag=10, criterion="bic",
                       method="lasso"):
    X, y =  form_Xy(X_raw=np.hstack((y[:, None], x[:, None])),
                    y_raw=y)
    return _compute_AR_error(X, y, criterion, method)


def multi_bivariate_AR_error(y, x_list, max_lag=10, criteroin="bic",
                             method="lasso"):
    return [bivariate_AR_error(y, x, max_lag=max_lag,
                               criterion=criterion, method=method)
                               for x in x_list]


def _compute_AR_error(X, y, criterion="bic", method="lasso"):
    if method == "lasso":
        w = estimate_b_lasso(X, y, criterion=criterion)
        return np.var(y - X @ w)
    elif method == "lstsqr":
        # With scipy
        # w = estimate_b_lstsqr(X, y)

        # linalg.lstsq
        # _, r, _, _ = lstsq(X, y, check_finite=False,
        #                    lapack_driver="gelsy")
        # return r / X.shape[0]

        # scipy.linalg cholesky solve -- by far the fastest
        w = linalg.cho_solve(linalg.cho_factor(X.T @ X), X.T @ y)
        return np.var(y - X @ w)
    else:
        raise ValueError("Method {} not available!".format(method))
    

def compute_pairwise_gc(X, max_lag=10, criterion="bic", method="lasso",
                        n_jobs=1):
    T, n = X.shape

    # NOTE: Since sklearn uses multiple cores, this method already does make
    # NOTE: reasonably efficient use of available processors.

    xi_i = np.array([univariate_AR_error(X[:, i], max_lag=max_lag,
                                         criterion=criterion, method=method)
                     for i in range(n)])

    # if n_jobs > 1:
    #     pool = Pool(2)
    #     # xi_i = np.array(
    #     #     pool.starmap(univariate_AR_error,
    #     #                  [(X[:, i], max_lag, criterion, method)
    #     #                   for i in range(n)]))
    #     xi_ij = np.array(pool.starmap(multi_bivariate_AR_error,
    #                                   [(X[:, i], [X[:, j] for j in range(n)],
    #                                     max_lag, criterion, method)
    #                                    for i in range(n)]))
    #     xi_ij[range(n), range(n)] = xi_i

    # else:
    xi_ij = [[bivariate_AR_error(X[:, i], X[:, j], max_lag=max_lag,
                                 criterion=criterion, method=method)
              if j != i else xi_i[i] for j in range(n)]
             for i in range(n)]

    F = -np.log(xi_ij / xi_i[:, None])
    F[F < 0] = 0.0
    return F


def pw_scg(F, delta=None, b=None, R=None):
    """
    Graph recovery heuristic

    F should be an n x n array where each entry is given by F_ij =
    log(xi_ij / xi_i) i.e. the normalized (pairwise) GC statistic.

    delta is a thresholding parameter, b is a "branching" parameter,
    and R is the number of root nodes.

    We return an nx.DiGraph

    Defaults: R = log(n)
              b = sqrt(n)
              delta = median(F)  # NOTE: this will often be 0
    """
    def arg_select_min_N(I, N):
        arg_sorted = sorted(I.keys(), key=lambda k: I[k])
        return set(arg_sorted[:N])

    def compute_incident_strength(F, S, W_pred):
        return {i: sum(F[i, j] for j in W_pred[i]) for i in S}

    def sort_edges_by_F(P_kr, P_k, F, W):
        return sorted([(i, j) for (i, j) in W if (i in P_kr and j in P_k)],
                      key=lambda ij: F[ij[1], ij[0]], reverse=True)

    def has_path(G, i, j):
        try:
            next(iter(nx.algorithms.simple_paths.all_simple_paths(G, i, j)))
        except StopIteration:
            return False
        else:
            return True

    n = F.shape[0]
    S = set(range(n))

    # These are really bad ways to choose parameters
    if R is None:
        R = ceil(np.log(n))
    if b is None:
        b = ceil(np.sqrt(n))
    if delta is None:
        delta = np.median(F)

    # Graph to return
    G = nx.DiGraph()
    G.add_nodes_from(S)

    # ------------- Initialization -------------
    # Set of candidate edges
    W_set = {(i, j) for (i, j) in product(S, S)
             if (F[j, i] > F[i, j] and F[j, i] > delta)}

    # Predecessor edges of a node, i.e. W[i] = {j | (j, i) \in W}
    W_pred = {i: [j for j in S if (j, i) in W_set ]
              for i in S}

    k = 1  # Counter purely for detecting infinite loops

    # incident strength of each node
    I = compute_incident_strength(F, S, W_pred)
    P = arg_select_min_N(I, N=R)
    P_k = P

    # ------------ Iterations -------------------
    while len(S) != 0:
        S = S - P_k
        I = compute_incident_strength(F, S, W_pred)
        P_k = arg_select_min_N(I, N=b)

        for i, j in sort_edges_by_F(P, P_k, F, W_set):
            if not any((has_path(G, a, j) for a in G.predecessors(i))):
                G.add_edge(i, j)
        P = P | P_k

        if k > 10 * n ** 2:  # Clearly stuck
            raise AssertionError("pw_scg has failed to terminate after {} iterations.  "
                                 "S = {}, P = {}".format(k, S, P))
        k = k + 1

    return G


def estimate_graph(X, criterion="bic", max_lags=10):
    F = compute_pairwise_gc(X, max_lag=max_lags,
                            criterion=criterion,
                            method="lstsqr")

    G_hat = pw_scg(F, R=None, b=None, delta=np.median(F))
    G_hat = attach_node_prop(G_hat, G, prop_attach="x", prop_from="x")
    G_hat = add_self_loops(G_hat, copy_G=False)
    G_hat = estimate_B(G_hat, max_lags, copy_G=False, max_T=T,
                       ic="bic")
    G_hat = remove_zero_filters(G_hat, "b_hat(z)", copy_G=False)
    return G_hat


def compute_tp_tn(N_edges, N_hat_edges, N_intersect_edges, n_nodes):
    P = N_edges  # Positives
    TP = N_intersect_edges  # True positives
    FP = N_hat_edges - N_intersect_edges  # False positives
    TPR = TP / P  # True positive rate (Sensitivity)

    N = n_nodes**2 - N_edges  # Negatives
    FN = N_edges - N_intersect_edges  # False negatives
    TN = N - FN  # True negatives
    TNR = TN / N  # True negative rate
    return TP, TN, FP, FN


def compute_F1_score(N_edges, N_hat_edges, N_intersect_edges, n_nodes):
    TP, TN, FP, FN = compute_tp_tn(N_edges, N_hat_edges, N_intersect_edges,
                                   n_nodes)

    F1 = (2 * TP) / (2 * TP + FP + FN)  # F1 score
    return F1


def compute_MCC_score(N_edges, N_hat_edges, N_intersect_edges, n_nodes):
    TP, TN, FP, FN = compute_tp_tn(N_edges, N_hat_edges, N_intersect_edges,
                                   n_nodes)
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) *
                                        (TN + FP) * (TN + FN))
    return MCC


def example_graph():
    N = list(range(1, 7))
    E = [(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]

    G = nx.DiGraph()
    G.add_nodes_from(N)
    G.add_edges_from(E)  # This will add nodes if they aren't already there.

    # set(G.predecessors(i)): set of parents of i in G
    # set(nx.ancestors(G, i)): set of ancestors of i in G
    # A = nx.adjacency_matrix(G).todense(): Find adj matrix
    return

