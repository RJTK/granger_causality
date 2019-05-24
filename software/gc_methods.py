import networkx as nx
import numpy as np

from sklearn.linear_model import LinearRegression, LassoLarsIC

from itertools import product
from math import ceil

def estimate_b_lstsqr(X, y):
    """
    Simple direct estimate of a linear filter y_hat = Xb.

    y: np.array (T)
    X: np.array (T x s)
    """
    lr = LinearRegression(fit_intercept=False, normalize=False,
                          copy_X=False)
    lr.fit(X, y)
    return lr.coef_


def estimate_b_lasso(X, y, criterion="bic"):
    """
    Estimate y_hat = Xb via LASSO and using BIC to choose regularizers.

    y: np.array (T)
    X: np.array (T x s)
    """
    lasso = LassoLarsIC(criterion=criterion, fit_intercept=False,
                        normalize=False)
    lr = LinearRegression(fit_intercept=False, normalize=False,
                          copy_X=False)

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
        a_i = list(G.predecessors(i))  # Inputs to i
        if len(a_i) == 0:
            continue

        # Form an "un-lagged" data matrix
        X_raw = np.hstack([G.nodes[j]["x"][:, None] for j in a_i])
        y_raw = G.nodes[i]["x"]

        # Form linear regression matrices
        X, y = form_Xy(X_raw, y_raw, p=max_lag)

        # Estimate
        b = estimate_b_lasso(X[:max_T], y[:max_T], criterion=ic)

        # Add the filters as properties to G
        for grouping, j in enumerate(a_i):  # Edge j --b(z)--> i
            G[j][i]["b_hat(z)"] = b[grouping * max_lag: (grouping + 1) * max_lag]
        G.nodes[i]["sv^2_hat"] = np.var(y - X @ b)
    return G


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
              delta = median(F)
    """
    def arg_select_min_N(I, N):
        arg_sorted = sorted(I.keys(), key=lambda k: I[k])
        return set(arg_sorted[:N])

    def compute_incident_strength(F, S):
        return {i: sum(F[i, j] for j in W_pred[i]) for i in S}

    def sort_edges_by_F(P_kr, P_k, F):
        return sorted(zip(P_kr, P_k), key=lambda ij: F[ij], reverse=True)

    def has_path(G, i, j):
        try:
            next(iter(nx.algorithms.simple_paths.all_simple_paths(G, i, j)))
        except StopIteration:
            return False
        else:
            return True

    n = F.shape[0]
    S = set(range(n))

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
         if F[j, i] > F[i, j]}

    # Predecessor edges of a node, i.e. W[i] = {j | (j, i) \in W}
    W_pred = {i: [j for j in S if (j, i) in W_set ]
              for i in S}

    # incident strength of each node
    I = compute_incident_strength(F, S)
    P = [arg_select_min_N(I, N=R)]  # P_0, P_1, ...

    # ------------ Iterations -------------------
    k = 1
    while len(S) != 0:
        S = S - P[k - 1]
        I = compute_incident_strength(F, S)
        P = P + [arg_select_min_N(I, N=b)]

        for r in range(1, k + 1):
            for i, j in sort_edges_by_F(P[k - r], P[k], F):
                if not has_path(G, i, j):
                    G.add_edge(i, j)
        k = k + 1

        if k > 10 * n ** 2:  # Clearly stuck
            raise AssertionError("pw_scg has failed to terminate after {} iterations.  "
                                 "S = {}, P = {}".format(k, S, P))
    
    return G


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
