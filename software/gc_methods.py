import networkx as nx
import numpy as np

from sklearn.linear_model import LinearRegression, LassoLarsIC

from itertools import product

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


def pw_scg(F, delta, b, R):
    """
    Graph recovery heuristic

    F should be an n x n array where each entry is given by F_ij =
    log(xi_ij / xi_i) i.e. the normalized (pairwise) GC statistic.

    delta is a thresholding parameter, b is a "branching" parameter,
    and R is the number of root nodes.
    """
    n = F.shape[0]

    # List of candidate edges
    W_set = {(i, j) for (i, j) in product(range(n), range(n))
         if F[j, i] > F[i, j]}

    S = list(range(n))
    I = []
    return


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
