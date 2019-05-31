import networkx as nx
import numpy as np
import sympy as sp

from scipy import linalg
from scipy.linalg import toeplitz
from sklearn.linear_model import LinearRegression, LassoLarsIC

from itertools import product
from math import ceil

from var_system import (attach_node_prop, add_self_loops,
                        remove_zero_filters, get_X)

def estimate_b_lstsqr(X, y):
    """
    Simple direct estimate of a linear filter y_hat = Xb.

    y: np.array (T)
    X: np.array (T x s)
    """
    w = linalg.cho_solve(linalg.cho_factor(X.T @ X), X.T @ y)
    return w


def estimate_b_lstsqr_cov(R, r):
    try:
        return linalg.cho_solve(linalg.cho_factor(R), r)
    except linalg.LinAlgError:
        # print("WARNING: R matrix is not PSD!  lambda_min(R) = {}"
        #       "".format(np.min(linalg.eigvals(R))))
        return linalg.solve(R, r, check_finite=True, lower=True)


def estimate_b_lasso(X, y, criterion="bic"):
    """
    Estimate y_hat = Xb via LASSO and using BIC to choose regularizers.

    y: np.array (T)
    X: np.array (T x s)
    """
    lasso = LassoLarsIC(criterion=criterion, fit_intercept=False,
                        normalize=False, precompute=True)

    # NOTE: This takes 5 - 10 ms for (5000, 125) matrix X
    lasso.fit(X, y)
    w = lasso.coef_
    if np.all(w == 0):
        # All zero support
        return w

    # Use lasso only to select the support
    X_lasso = X[:, [i for i, wi in enumerate(w) if wi != 0]]
    w_lr = estimate_b_lstsqr(X_lasso, y)
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


def block_toeplitz(left_col, top_row=None):
    p = len(left_col)
    assert len(top_row) == p
    assert np.allclose(left_col[0], top_row[0])
    if top_row is None:
        top_row = left_col

    f = left_col + top_row[::-1][:-1]

    # return np.block([top_list] +
    #                 [left_list[tau::-1] +
    #                  top_list[tau:] for tau in range(1, p)])
    return np.block([[f[i - j] for j in range(p)] for i in range(p)])


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

        # Compute residuals
        r = y - X @ b

        # Add the filters as properties to G
        for grouping, j in enumerate(a_i):  # Edge j --b(z)--> i
            G[j][i]["b_hat(z)"] = b[grouping * max_lag: (grouping + 1) * max_lag]

        # As well as the residuals
        G.nodes[i]["sv^2_hat"] = np.var(r)
        G.nodes[i]["r"] = r
    return G


# NOTE: forming the Xy matrices consumes only microseconds
def univariate_AR_error(x, max_lag=10, criterion="bic",
                        method="lasso"):
    X, y = form_Xy(X_raw=x[:, None], y_raw=x, p=max_lag)
    return _compute_AR_error(X, y, criterion, method)


def bivariate_AR_error(y, x, max_lag=10, criterion="bic",
                       method="lasso"):
    X, y =  form_Xy(X_raw=np.hstack((y[:, None], x[:, None])),
                    y_raw=y, p=max_lag)
    return _compute_AR_error(X, y, criterion, method)


def compute_covariances(X, p, symmetrize=False):
    T = X.shape[0]
    R = np.dstack(
        [X[p:, :].T @ X[p:, :]] + 
        [X[p:, :].T @ X[p - tau: -tau, :]
         for tau in range(1, p + 1)]) / (T - p + 1)
    return R


def _compute_AR_error(X, y, criterion="bic", method="lasso"):
    if method == "lasso":
        w = estimate_b_lasso(X, y, criterion=criterion)
        return np.var(y - X @ w)
    elif method == "lstsqr":
        w = estimate_b_lstsqr(X, y)
        return np.var(y - X @ w)
    else:
        raise ValueError("Method {} not available!".format(method))
    

def _fast_compute_AR_error(R, r):
    w = estimate_b_lstsqr_cov(R, r)
    return max(0, R[0, 0] - w @ r)


def compute_gc_score(xi_i, xi_ij):
    F = -np.log(xi_ij / xi_i[:, None])
    F[F < 0] = 0.0
    F[np.isnan(F)] = 0.0
    return F


def compute_xi(X, max_lag, criterion="bic", method="lstsqr"):
    n = X.shape[1]
    xi_i = np.array([univariate_AR_error(X[:, i], max_lag=max_lag,
                                         criterion=criterion, method=method)
                     for i in range(n)])
    xi_ij = np.array([[bivariate_AR_error(X[:, i], X[:, j], max_lag=max_lag,
                                          criterion=criterion, method=method)
                       if j != i else xi_i[i] for j in range(n)]
                      for i in range(n)])
    return xi_i, xi_ij


def fast_compute_xi(X, max_lag=10, reg_delta=0.0):
    T, n = X.shape
    R = compute_covariances(X, max_lag)

    xi_i = np.array([_fast_compute_AR_error(
        toeplitz(R[i, i, :-1]), R[i, i, 1:])
                     for i in range(n)])

    # TODO: Most of the time is now spent forming toeplitz matrices.

    # TODO:
    # due to inaccurate estimates of covariance, these matrices
    # are not actually guaranteed to be positive definite.
    # We fallback on linalg.solve in those cases, but it might
    # be reasonable instead to fallback onto form_Xy and direct
    # least squares.
    def form_Rrij(R, i, j):
        C = toeplitz(R[j, i, :-1], R[i, j, :-1])  # Left column then top row!
        Rij = np.block([[toeplitz(R[i, i, :-1]), C],
                        [C.T, toeplitz(R[j, j, :-1])]])
        rij = np.concatenate((R[i, i, 1:], R[i, j, 1:]))
        Rij = Rij + reg_delta * np.eye(Rij.shape[0])
        return Rij, rij

    xi_ij = np.array([[_fast_compute_AR_error(*form_Rrij(R, i, j))
                       if j != i else xi_i[i] for j in range(n)]
                      for i in range(n)])
    return xi_i, xi_ij


def compute_pairwise_gc(X, max_lag=10, criterion="bic", method="lasso"):
    xi_i, xi_ij = compute_xi(X, max_lag, criterion, method)
    return compute_gc_score(xi_i, xi_ij)


def fast_compute_pairwise_gc(X, max_lag=10):
    """
    This is /dramatically/ faster than compute_pairwise_gc,
    only difference is that it handles just plain least squares.

    TODO: However there are significant discrepancies!
    TODO: There is something clearly wrong about this implementation.
    """
    xi_i, xi_ij = fast_compute_xi(X, max_lag)
    return compute_gc_score(xi_i, xi_ij)


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

    def get_paths(G, i, j):
        return list(nx.algorithms.simple_paths.all_simple_paths(G, i, j))

    def is_strongly_causal(G):
        for j in G.nodes:
            for i in nx.ancestors(G, j):
                num_paths = len(get_paths(G, i, j))
                if num_paths == 0:
                    raise AssertionError("{} is not an ancestor of {}!"
                                         "".format(i, j))
                elif num_paths > 1:
                    return False
        return True

    def maintains_strong_causality(G, i, j):
        """
        Whether adding edge i, j to G maintains strong causality.
        """
        common_ancestors = len(nx.ancestors(G, i) & nx.ancestors(G, j))
        common_descendants = len(nx.descendants(G, i) & nx.descendants(G, j))
        return common_ancestors == 0 and common_descendants == 0

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
    P_k = [P]

    # ------------ Iterations -------------------
    while len(S) != 0:
        S = S - P_k[-1]
        I = compute_incident_strength(F, S, W_pred)
        P_k_next = arg_select_min_N(I, N=b * len(P_k[-1]))
        P_k.append(P_k_next)

        for r in range(1, k + 1):
            for i, j in sort_edges_by_F(P_k[-1 - r], P_k[-1], F, W_set):
                if maintains_strong_causality(G, i, j):
                    G.add_edge(i, j)
        P = P | P_k[-1]

        if k > 10 * n ** 2:  # Clearly stuck
            raise AssertionError("pw_scg has failed to terminate after {} iterations.  "
                                 "S = {}, P = {}".format(k, S, P))
        k = k + 1

    assert is_strongly_causal(G), "G is not strongly causal!"
    return G


def full_filter_estimator(G, criterion="bic", max_lags=10,
                          M_passes=1, T_max=None):
    # This can result in extremely long lag lengths
    # when the final filter is built.

    G_hats = []
    G_hat = G.copy()
    if T_max is None:
        T_max = len(G.nodes[0]["x"])

    for m in range(M_passes):
        X = get_X(G_hat)[:T_max]
        G_hat = estimate_graph(X, G_hat, criterion, max_lags)
        G_hats.append(G_hat)
        G_hat = get_residual_graph(G_hat.copy())

    # G_hat = construct_complete_filter(G_hats)
    G_hat = combine_graphs(G_hats)
    G_hat = attach_node_prop(G_hat, G, "x", "x")
    G_hat = estimate_B(G_hat, max_lags, ic="bic")
    G_hat = remove_zero_filters(G_hat, "b_hat(z)", copy_G=False)
    return G_hat


def combine_graphs(G_hat_list):
    G_hat = nx.DiGraph()
    G_hat.add_nodes_from(G_hat_list[0].nodes)

    for G_i in G_hat_list:
        G_hat.add_edges_from(list(G_i.edges))
    return G_hat


# def construct_complete_filter(G_hat_list):
#     z = sp.Symbol("z")
#     filters = []
#     n = len(G_hat_list[0].nodes)

#     def construct_filter(G_hat):
#         var = gcg_to_var(G_hat, "b_hat(z)")
#         B = sp.Matrix(np.zeros((n, n)))

#         B_list = var.B
#         for tau, B_tau in enumerate(B_list):
#             B = B + (z ** (tau + 1)) * sp.Matrix(B_tau)
#         return B

#     B = construct_filter(G_hat_list[-1])
#     for G_hat in G_hat_list[:-1]:
#         B_i = sp.Matrix(np.eye(n)) - construct_filter(G_hat)
#         B = B * B_i
#     B = sp.expand(B)

#     def get_poly_coefs(b):
#         try:
#             return b.as_poly().all_coeffs()[::-1]
#         except sp.GeneratorsNeeded:
#             return [0]

#     def get_B(coefs, tau):
#         n = len(coefs)
#         B = np.zeros((n, n))
#         for i in range(n):
#             for j in range(n):
#                 bij = coefs[i][j]
#                 try:
#                     B[i, j] = bij[tau]
#                 except IndexError:
#                     pass
#         return B

#     coef_lists = [[get_poly_coefs(B[i, j]) for j in range(n)]
#                   for i in range(n)]
#     # TODO: Can actually create G directly from here.

#     p_max = max(map(max, [map(len, coefs) for coefs in coef_lists]))
#     B_list = [get_B(coef_lists, tau) for tau in range(p_max)]
#     B = np.dstack(B_list)

#     G_hat = nx.DiGraph()

#     for i in range(n)
#     return B

def get_residual_graph(G_hat):
    """
    Essentially just replaces node properties "r" with "x".
    """
    for i in G_hat.nodes:
        G_hat.nodes[i]["x"] = G_hat.nodes[i]["r"]
        del G_hat.nodes[i]["r"]
    return G_hat


def estimate_graph(X, G, criterion="bic", max_lags=10):
    """
    Produce an estimated graph from the data X.

    This function suffers from poor design as I am also needing to
    pass in a "true" graph G in order to attach the right data to
    G_hat and calculate the estimate of sigma_v^2.

    TODO: This probably needs to be fixed before doing any application
    """
    F = compute_pairwise_gc(X, max_lag=max_lags,
                            criterion=criterion,
                            method="lstsqr")

    # F = fast_compute_pairwise_gc(X, max_lag=max_lags)


    T, n = X.shape
    G_hat = pw_scg(F, R=1 + int(np.log(n)), b=2, delta=0.05)
    G_hat = attach_node_prop(G_hat, G, prop_attach="x", prop_from="x")
    G_hat = add_self_loops(G_hat, copy_G=False)
    # draw_graph_estimates(G, G_hat)

    G_hat = estimate_B(G_hat, max_lags, copy_G=False, max_T=X.shape[0],
                       ic=criterion)
    G_hat = remove_zero_filters(G_hat, "b_hat(z)", copy_G=False)
    # draw_graph_estimates(G, G_hat)
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

