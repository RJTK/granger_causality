"""
Methods which implement granger-causality computations.
"""

import networkx as nx
import numpy as np
import numba
from levinson import (lev_durb, whittle_lev_durb)

from scipy import linalg
from scipy import stats as sps
from scipy.linalg import toeplitz
from sklearn.linear_model import LassoLarsIC

from itertools import product
from functools import reduce
from math import ceil

# TODO: What is the proper way to do this?  I want packaging,
# TODO: but I also want to be able to C-c C-c this file into
# TODO: ipython while I am working...
try:
    from .stat_util import benjamini_hochberg
    from .var_system import (attach_node_prop, add_self_loops,
                             remove_zero_filters, get_X,
                             make_complete_digraph, attach_X)
except ImportError:
    from stat_util import benjamini_hochberg
    from var_system import (attach_node_prop, add_self_loops,
                            remove_zero_filters, get_X,
                            make_complete_digraph, attach_X)


def estimate_b_lstsqr(X, y):
    """
    Simple direct estimate of a linear filter y_hat = Xb.

    y: np.array (T)
    X: np.array (T x s)
    """
    w = linalg.cho_solve(linalg.cho_factor(X.T @ X), X.T @ y)
    return w


def estimate_b_lstsqr_cov(R, r):
    return linalg.cho_solve(linalg.cho_factor(R), r)


def estimate_b_lasso(X, y):
    """
    Estimate y_hat = Xb via LASSO and using BIC to choose regularizers.

    y: np.array (T)
    X: np.array (T x s)
    """
    lasso = LassoLarsIC(criterion="bic", fit_intercept=False,
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
        [np.hstack([X_raw[p - tau: -tau, j][:, None]
                    for tau in range(1, p + 1)])
         for j in range(s)])

    assert X.shape == (T - p, s * p), "Wrong _X shape!"
    assert len(y) == T - p, "wrong y shape!"
    return X, y


def block_toeplitz(left_col, top_row=None):
    """
    Similarly to linalg.toeplitz but for blocks.
    """
    p = len(left_col)
    left_col = np.array(left_col)  # In case a list is passed in
    if top_row is None:
        top_row = [left_col[0]]
        top_row = np.array(top_row +
                           [np.transpose(np.conj(left_col[k]))
                            for k in range(1, p)])
    assert len(top_row) == p
    assert np.allclose(left_col[0], top_row[0])

    try:
        f = np.vstack((left_col, top_row[::-1][:-1]))
    except ValueError:
        # When the arrays are unidimensional
        f = np.hstack((left_col, top_row[::-1][:-1]))
    return np.block([[f[i - j] for j in range(p)] for i in range(p)])


def estimate_B(G, max_lag=10, copy_G=False,
               max_T=None, method="lasso"):
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
        if method == "lasso":
            b = estimate_b_lasso(X[:max_T], y[:max_T])
        elif method == "lstsqr":
            b = estimate_b_lstsqr(X[:max_T], y[:max_T])
        else:
            raise AssertionError("Bad method but should deal with it earlier!")

        # Compute residuals
        r = y[max_T:] - X[max_T:] @ b

        # Add the filters as properties to G
        for grouping, j in enumerate(a_i):  # Edge j --b(z)--> i
            G[j][i]["b_hat(z)"] = b[grouping * max_lag:
                                    (grouping + 1) * max_lag]

        # As well as the residuals
        G.nodes[i]["sv^2_hat"] = np.var(r)
        G.nodes[i]["r"] = r
    return G


# NOTE: forming the Xy matrices consumes only microseconds
# NOTE: But is the majority of time spent...
def univariate_AR_error(x, max_lag=10):
    # Try larger model orders until we don't increase the BIC
    bic = -np.infty
    T = len(x)
    for p in range(1, max_lag + 1):
        X_, y_ = form_Xy(X_raw=x[:, None], y_raw=x, p=p)
        sv2_p = _compute_AR_error(X_, y_)
        bic_p = -T * np.log(sv2_p) - p * np.log(T)
        if bic_p > bic:
            bic = bic_p
            continue
        else:
            p = p - 1
            break
    return sv2_p, p


def bivariate_AR_error(y, x, max_lag=10):
    bic = -np.infty
    T = len(x)
    for p in range(1, max_lag + 1):
        X_, y_ = form_Xy(X_raw=np.hstack((y[:, None], x[:, None])),
                         y_raw=y, p=p)
        sv2_p = _compute_AR_error(X_, y_)
        bic_p = -T * np.log(sv2_p) - 4 * p * np.log(T)
        if bic_p > bic:
            bic = bic_p
            continue
        else:
            p = p - 1
            break
    return sv2_p, p


@numba.jit(nopython=True, cache=True)
def compute_covariances(X, p):
    # Recall that we must ensure the result is positive definite --
    # calculate the cov of a windowed signal.
    T, n = X.shape
    R = np.empty((p + 1, n, n))
    X = X / np.sqrt(T)
    R[0] = X.T @ X
    for tau in range(1, p + 1):
        R[tau] = X[tau:, :].T @ X[:-tau, :]
    return R


def _compute_AR_error(X, y):
    w = estimate_b_lstsqr(X, y)
    return np.var(y - X @ w)


def _fast_compute_AR_error(R, r):
    w = estimate_b_lstsqr_cov(R, r)
    return max(0, R[0, 0] - w @ r)


@numba.jit(nopython=True, cache=True)
def _fast_univariate_AR_error(r, T):
    _, _, eps = lev_durb(r)
    bic = compute_bic(eps, T, s=1)
    p_opt = np.argmax(bic)
    return eps[p_opt], p_opt


@numba.jit(nopython=True, cache=True)
def _fast_bivariate_AR_error(R, T):
    _, _, S = whittle_lev_durb(R)
    det_S = np.empty(len(S))
    for k in range(len(S)):
        det_S[k] = np.linalg.det(S[k])
    bic = compute_bic(det_S, T, s=4)
    p_opt = np.argmax(bic)
    S_opt = S[p_opt]
    return S_opt[0, 0], S_opt[1, 1], p_opt


@numba.jit(nopython=True, cache=True)
def compute_bic(eps, T, s=1):
    """
    set s = 1 for univariate case and s = 4 for bivariate case.
    """
    bic = np.empty(len(eps))
    logT = np.log(T)
    for p, log_eps_p in enumerate(np.log(eps)):
        bic[p] = -log_eps_p - s * p * logT / T
    return bic


def compute_gc_score(xi_i, xi_ij, T, p_lags):
    F = T * ((xi_i[:, None] / xi_ij) - 1) / p_lags
    F[p_lags == 0] = 0
    return F


@numba.jit(nopython=True, cache=True)
def form_bivariate_covariance(R, i, j):
    """
    Creates bivariate covariance sequence by pulling out R[:, i, j]
    components.
    """
    # return R[:, [i, j], :][:, :, [i, j]]  # Only basic indexing for numba
    p, _, _ = R.shape
    Rij = np.empty((p, 2, 2))
    Rij[:, 0, 0] = R[:, i, i]
    Rij[:, 0, 1] = R[:, i, j]
    Rij[:, 1, 0] = R[:, j, i]
    Rij[:, 1, 1] = R[:, j, j]
    return Rij


def compute_xi(X, max_lag):
    """
    Calculates xi_i and xi_ij, the errors for uni- and bi- variate
    AR models.  We return xi_i (array), xi_ij (matrix), p_i (array),
    p_ij (matrix)

    p_i and p_ij are lag lengths estimated via BIC.
    """
    n = X.shape[1]
    xi_p_i = np.array(
        [univariate_AR_error(X[:, i], max_lag=max_lag)
         for i in range(n)])
    xi_i, p_i = xi_p_i[:, 0], xi_p_i[:, 1]

    xi_p_ij = np.array(
        [[bivariate_AR_error(X[:, i], X[:, j], max_lag=max_lag)
          if j != i else xi_p_i[i] for j in range(n)]
         for i in range(n)])
    xi_ij, p_ij = xi_p_ij[:, :, 0], xi_p_ij[:, :, 1]
    return xi_i, xi_ij, p_i, p_ij


# Numba parallelism seems much slower unfortunately
# @numba.jit(nopython=True, cache=True, parallel=True)
@numba.jit(nopython=True, cache=True)
def fast_compute_xi(X, max_lag=10):
    T, n = X.shape
    R = compute_covariances(X, max_lag)

    xi_i = np.zeros(n)
    xi_ij, p_ij = np.zeros((n, n)), np.zeros((n, n))

    for i in range(n):
        xi_i[i], p_i = _fast_univariate_AR_error(R[:, i, i], T)
        xi_ij[i, i] = xi_i[i]
        p_ij[i, i] = p_i

    for i in numba.prange(n):
        for j in range(i):
            Rij = form_bivariate_covariance(R, i, j)
            ei, ej, _p = _fast_bivariate_AR_error(Rij, T)
            xi_ij[i, j] = ei
            xi_ij[j, i] = ej
            p_ij[i, j] = _p
            p_ij[j, i] = _p
    return xi_i, xi_ij, p_ij


# TODO: Use the criterion for selecting lag lengths
# TODO: There are actually a few ways I could determine edge prescense
# TODO: (1) for scarce data, use sklearn ARD or possibly LASSO
# TODO: (2) when data is abundant stick with OLS and chi2 tests
# TODO: -- However, still need to select number of parameters!
def compute_pairwise_gc(X, max_lag=10):
    T, _, _ = *X.shape, max_lag
    xi_i, xi_ij, p_i, p_ij = compute_xi(
        X, max_lag)
    F = compute_gc_score(xi_i, xi_ij, T, p_ij)
    return F, p_ij  # p_i is actually irrelevant.


def normalize_gc_score(F, p):
    F = sps.chi2.cdf(F, p)
    return F


def fast_compute_pairwise_gc(X, max_lag=10):
    """
    This is /dramatically/ faster than compute_pairwise_gc.

    TODO: However there are significant discrepancies!
    TODO: There is something clearly wrong about this implementation.
    """
    T, _, = X.shape
    xi_i, xi_ij, p_ij = fast_compute_xi(X, max_lag)
    return compute_gc_score(xi_i, xi_ij, T, p_ij), p_ij


# TODO: P_j should be an n x n array
# TODO: Tune alpha by Benjamini Hochberg criterion
def pw_scg(F, P_edge, alpha):
    """
    Graph recovery heuristic

    F should be an n x n array where each entry is given by

    F_ij = (T / p_j) * [(xi_i / xi_ij) - 1] i.e. chi2(p_j) test statistic.

    P_edge is the edge probabilities (i.e. 1 - P_value) with the same
    shape as F, and alpha is a test level (i.e. from
    Benjamini-Hochberg) s.t. we will say an edge is present if P_edge
    >= 1 - alpha.

    We return an nx.DiGraph
    """
    def arg_select_min_N(I, N):
        arg_sorted = sorted(I.keys(), key=lambda k: I[k])
        return set(arg_sorted[:N])

    def compute_incident_strength(F, S, W_pred):
        return {i: sum(F[i, j] for j in S & set(W_pred[i])) for i in S}

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
        if i == j:
            return False
        anc_j = nx.ancestors(G, j)
        anc_i = nx.ancestors(G, i)
        if (i in anc_j) or (j in anc_i):
            return False
        if len(anc_i & anc_j):
            # Common ancestors
            return False
        if len(nx.descendants(G, i) & nx.descendants(G, j)):
            # Common descendants
            return False
        return True

    n = F.shape[0]
    S = set(range(n))
    # Larger value means more confident about an edge
    P_edge[P_edge < 1 - alpha] = 0

    # Graph to return
    G = nx.DiGraph()
    G.add_nodes_from(S)

    # ------------- Initialization -------------
    # Set of candidate edges
    W_set = {(i, j) for (i, j) in product(S, S)
             if (F[j, i] >= F[i, j] and P_edge[j, i] > 0)}

    # Predecessor edges of a node, i.e. W[i] = {j | (j, i) \in W}
    W_pred = {i: [j for j in S if (j, i) in W_set]
              for i in S}

    k = 1  # Counter purely for detecting infinite loops

    # incident strength of each node
    # I = compute_incident_strength(F, S, W_pred)

    # Probability sum of input edges  (roughly an incident edge count)
    C = compute_incident_strength(P_edge, S, W_pred)

    # Choose the next layer based on the number of incident edges,
    # and ensure that we choose at least 1.
    P = arg_select_min_N(C, N=sum(C[i] < ceil(min(C.values())) for i in C))
    if len(P) == 0:
        P = arg_select_min_N(
            C, N=sum(C[i] <= ceil(min(C.values())) for i in C))
    P_k = [P]

    # ------------ Iterations -------------------
    while len(S) != 0:
        S = S - P_k[-1]
        # I = compute_incident_strength(F, S, W_pred)
        C = compute_incident_strength(P_edge, S, W_pred)

        P_k_next = arg_select_min_N(
            C, N=sum(C[i] < ceil(min(C.values())) for i in C))
        if len(P_k_next) == 0:
            P_k_next = arg_select_min_N(
                C, N=sum(C[i] <= ceil(min(C.values())) for i in C))

        P_k.append(P_k_next)

        # Append new edges in order of test statistic magnitude
        # for r in range(1, k + 1):
        # NOTE: It appears that sorting on F on all of the previous P_k sets
        # NOTE: works pretty well.  We should expect that "real" edges will
        # NOTE: have larger F values.  In the "theoretical" algorithm, we can
        # NOTE: only check yes/no, for that reason we need to apply backwards
        for i, j in sort_edges_by_F(reduce(
                set.union, P_k[:-1]), P_k[-1], F, W_set):
            if maintains_strong_causality(G, i, j):
                G.add_edge(i, j)
        P = P | P_k[-1]

        if k > 10 * n ** 2:  # Clearly stuck
            raise AssertionError("pw_scg has failed to terminate after {} "
                                 "iterations.  "
                                 "S = {}, P = {}".format(k, S, P))
        k = k + 1

    assert is_strongly_causal(G), "G is not strongly causal!"
    return G


def full_filter_estimator(G, max_lags=10,
                          M_passes=1, T_max=None):
    # This can result in extremely long lag lengths
    # when the final filter is built.

    raise NotImplementedError("Review this -- I also think it's a bad idea.")

    G_hats = []
    G_hat = G.copy()
    if T_max is None:
        T_max = len(G.nodes[0]["x"])

    for m in range(M_passes):
        X = get_X(G_hat)[:T_max]
        G_hat = estimate_graph(X, G_hat, max_lags=max_lags)
        G_hats.append(G_hat)
        G_hat = get_residual_graph(G_hat.copy())

    # G_hat = construct_complete_filter(G_hats)
    G_hat = combine_graphs(G_hats)
    G_hat = attach_node_prop(G_hat, G, "x", "x")
    G_hat = estimate_B(G_hat, max_lags)
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


def estimate_dense_graph(X, max_lag=10,
                         max_T=None, method="lasso"):
    _, n = X.shape
    G = make_complete_digraph(n)
    G = attach_X(G, X)
    assert np.all(X == get_X(G))

    estimate_B(G, max_lag, copy_G=False, max_T=max_T, method=method)
    G = remove_zero_filters(G, "b_hat(z)", copy_G=False)
    return G


def estimate_graph(X, G, max_lags=10, method="lasso", alpha=0.05):
    """
    Produce an estimated graph from the data X.

    This function suffers from poor design as I am also needing to
    pass in a "true" graph G in order to attach the right data to
    G_hat and calculate the estimate of sigma_v^2.

    TODO: This probably needs to be fixed before doing any application
    """
    T, n = X.shape

    # Compute the pairwise errors and filter sizes
    # F, P = compute_pairwise_gc(X, max_lag=max_lags)
    F, P = fast_compute_pairwise_gc(X, max_lag=max_lags)

    # Screen edges via benjamini hochberg criterion
    P_edges = normalize_gc_score(F, P)  # p-values are just 1 - F
    P_values = 1 - P_edges[~np.eye(n, dtype=bool)].ravel()
    t_bh = benjamini_hochberg(P_values, alpha=alpha, independent=False)

    # Estimate a strongly causal graph
    G_hat = pw_scg(F, P_edges, t_bh)
    G_hat = attach_node_prop(G_hat, G, prop_attach="x", prop_from="x")
    G_hat = add_self_loops(G_hat, copy_G=False)

    # Restimate the coefficients
    if method == "lasso":
        G_hat = estimate_B(G_hat, max_lags, copy_G=False, max_T=T,
                           method=method)
        G_hat = remove_zero_filters(G_hat, "b_hat(z)", copy_G=False)
    elif method == "lstsqr":
        G_hat = estimate_B(G_hat, max_lags, copy_G=False, max_T=T,
                           method=method)
    else:
        raise NotImplementedError("The fitting method {} is not available!"
                                  "".format(method))
    return G_hat


def compute_tp_tn(N_edges, N_hat_edges, N_intersect_edges, n_nodes):
    # P = N_edges  # Positives
    TP = N_intersect_edges  # True positives
    FP = N_hat_edges - N_intersect_edges  # False positives
    # TPR = TP / P  # True positive rate (Sensitivity)

    N = n_nodes**2 - N_edges  # Negatives
    FN = N_edges - N_intersect_edges  # False negatives
    TN = N - FN  # True negatives
    # TNR = TN / N  # True negative rate
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
