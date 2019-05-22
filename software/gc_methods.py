import networkx as nx
import numpy as np

from sklearn.linear_model import LinearRegression, LassoLarsIC


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


def form_Xy(X, i, a, max_lag=10):
    """
    Builds appropriate (X, y) matrices for autoregressive prediction.

    X: np.array (T, n)
    i: index of X to model
    a: length n boolean array indicating which variables to include
      - denote s = sum(a) i.e. "sparsity"
    max_lag: the number of lagged values to include in AR predictions.

    returns: y_i, _X
    y_i: np.array (T - max_lag, 1)
    _X: np.array (T - max_lag, max_lag * s)
    """
    T, s = X[:, a].shape
    p = max_lag

    y = X[p:, i]
    _X = np.hstack([X[p - tau: -tau, a] for tau in range(1, p + 1)])
    assert _X.shape == (T - max_lag, s * max_lag)
    assert len(y) == T - max_lag
    return _X, y


def estimate_B(X, G, max_lag=10):
    """
    Estimate x(t) = B(z)x(t) respecting the topology of G.

    G: nx.DiGraph
    X: np.array (T x n)
    """
    T, n = X.shape

    # Sparsity pattern for B(z)
    A = np.array(
        nx.adjacency_matrix(G)\
        .todense())\
        .T\
        .astype(bool)
    assert A.shape == (n, n), "Wrong adjacency matrix shape!"
    A[range(n), range(n)] = True  # Always include self-loops

    for i in range(n):
        a_i = A[i, :]
        X_i, y_i = form_Xy(X, i, a_i, max_lag=max_lag)
        b = estimate_b_lasso(X_i, y_i)
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
