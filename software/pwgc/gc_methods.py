"""
Methods which implement granger-causality computations.
"""

import networkx as nx
import numpy as np
import numba

from multiprocessing import Pool
from itertools import product
from functools import reduce
from math import ceil

from scipy import linalg
from scipy import stats as sps
from scipy.linalg import toeplitz, cho_solve, cho_factor
from sklearn.linear_model import LassoLarsIC, LassoLarsCV, Lasso
from sklearn.preprocessing import StandardScaler
from spams import fistaFlat
from ts_lasso.ts_lasso import fit_VAR

from levinson import (lev_durb, whittle_lev_durb)

# TODO: What is the proper way to do this?  I want packaging,
# TODO: but I also want to be able to C-c C-c this file into
# TODO: ipython while I am working...
try:
    from .stat_util import benjamini_hochberg
    from .var_system import (attach_node_prop, add_self_loops,
                             remove_zero_filters, get_X,
                             make_complete_digraph, attach_X,
                             gcg_to_var)
except ImportError:
    from stat_util import benjamini_hochberg
    from var_system import (attach_node_prop, add_self_loops,
                            remove_zero_filters, get_X,
                            make_complete_digraph, attach_X,
                            gcg_to_var)


def estimate_b_lstsqr(X, y, ret_G=False, lmbda=np.inf):
    """
    Simple direct estimate of a linear filter y_hat = Xb.

    - if ret_G=True, we will return the gram Matrix X.T @ X.
    - lmbda is a regularzation term interpretable as the
      prior variance on the coefficient terms; i.e. we will
      compute the Tikhonov-regularized Leastsquares solution
      using X.T @ X / T + (1. / lmbda * T) as the covariance
      estimate.

    y: np.array (T)
    X: np.array (T x s)
    """
    T, n = X.shape
    R = (X.T @ X) / T
    r = X.T @ y / T

    R_reg = R + (1. / (lmbda * T)) * np.eye(n)
    try:
        w = cho_solve(cho_factor(R_reg), r)
    except linalg.LinAlgError:
        R_reg = R_reg + np.min(np.diag(R_reg)) *  np.eye(R_reg.shape[0])

    try:
        w = cho_solve(cho_factor(R_reg), r)
    except linalg.LinAlgError:
        w, _, _, _ = linalg.lstsq(R_reg, r)

    if ret_G:
        return w, R * T
    else:
        return w


def estimate_b_lstsqr_cov(R, r):
    return cho_solve(cho_factor(R), r)


def _estimate_b_lasso(G, X, y):
    """
    Lasso estimation given a precomputed Gram matrix G.
    """
    # NOTE: the 'normalize' parameter is ignored when fit_intercept=False
    lasso = LassoLarsIC(criterion="bic", fit_intercept=False,
                        normalize=False, precompute=G)
    w = lasso.fit(X, y).coef_
    return w


def _standardize_X(f):
    def _f(X, *args, **kwargs):
        _X = X - np.mean(X, axis=0)[None, :]
        s = np.std(_X, axis=0)[None, :]
        _X = _X / s

        w = f(_X, *args, **kwargs)
        w = w / s.ravel()
        return w
    return _f


@_standardize_X
def estimate_b_lasso(X, y):
    """
    Estimate y_hat = Xb via LASSO and using BIC to choose regularizers.

    y: np.array (T)
    X: np.array (T x s)
    """
    G = X.T @ X
    w = _estimate_b_lasso(G, X, y)
    return w


@_standardize_X
def estimate_b_alasso(X, y, lmbda_lstsqr=2.0, nu=2.0):
    """
    Estimate y_hat = Xb via Adaptive-LASSO
    and using BIC to choose regularizer.

    This is an adaptive scheme that uses "piloting weights" to improve
    the selection capacity of the lasso.

    lmbda_lstsqr: Prior variance for the lstsqr estimate (set to
      np.inf to disable)
    nu: exponential coefficient for the weights, larger values will
      put a greater emphasis on the lstsqr estimate, smaller values
      bring us close to the vanilla lasso.  Values larger than 1 can
      lead to substantial variability in the low sample regime,
      whereas nu > 1 allows for the LASSO regularizer to converge to
      0.  So for small samples: 0 < nu < 1 (i.e. nu = 0.5) and for
      larger samples nu > 1 (i.e. nu = 3/2).

    The oracle property requires that lmbda_n / sqrt(n) -> 0 and
    (lambda_n * n^(nu - 1)/2) -> oo

    y: np.array (T)
    X: np.array (T x s)
    """
    # NOTE: I am putting a regularizer on the cov matrix to avoid crazy
    # NOTE: variance in small samples, but inspection of the lstsqr
    # NOTE: function will show that my normalization is s.t.
    # NOTE: the estimator is still root(n)-consistent.
    b0, G = estimate_b_lstsqr(X, y, ret_G=True, lmbda=lmbda_lstsqr)
    wgt = np.abs(b0)**nu
    X_wgt = X * wgt[None, :]
    G_wgt = wgt[:, None] * G * wgt[None, :]

    w = _estimate_b_lasso(G_wgt, X_wgt, y)
    w = w * wgt
    return w


def estimate_b_glasso(X, y, p, T_train):
    # lmbda_g: Group Lasso regularization
    # lmbda_l: Lasso regularization

    _, _np = X.shape
    n = _np // p

    X_train, y_train = X[:T_train, :], y[:T_train, None]
    X_test, y_test = X[T_train:, :], y[T_train:, None]

    w0 = np.zeros((n * p, 1))

    eps = 1e-9
    lmbda_g = 1.0
    # lmbda_l = 1.0

    _X = np.asfortranarray(X_train)
    _y = np.asfortranarray(y_train)
    _w0 = np.asfortranarray(w0)

    err_star = np.inf
    w_star = _w0
    w = _w0
    for lmbda_g in np.logspace(-2, 2, 100):
        w, _ = fistaFlat(_y, _X, w,
                         loss="square", regul="group-lasso-l2",
                         lambda1=lmbda_g,
                         # lambda2=lmbda_l,
                         size_group=p, intercept=False, pos=False,
                         numThreads=4, max_it=100, subgrad=False,
                         return_optim_info=True)
        err = np.var(y_test - X_test @ np.array(w))
        if err < err_star:
            err_star = err
            w_star = w

    w_star = np.array(w_star)
    w_star[w_star <= eps] = 0.0
    return w_star


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
            G.nodes[i]["r"] = G.nodes[i]["x"][max_T + max_lag:]
            continue
        else:
            X_raw = np.hstack([G.nodes[j]["x"][:, None] for j in a_i])

        # Form linear regression matrices
        X, y = form_Xy(X_raw, y_raw, p=max_lag)

        X_maxT, y_maxT = X[:max_T], y[:max_T]

        # Estimate
        if method == "lasso":
            b = estimate_b_lasso(X_maxT, y_maxT)
        elif method == "glasso":
            b = estimate_b_glasso(X_maxT, y_maxT, p=max_lag, T_train=max_T)
        elif method == "alasso":
            b = estimate_b_alasso(X_maxT, y_maxT, nu=1.5)
        elif method == "lstsqr":
            b = estimate_b_lstsqr(X_maxT, y_maxT)
        else:
            raise AssertionError("Bad method but should deal with it earlier!")

        # if len(y[max_T:]) <= 100:
        #     raise ValueError(
        #         "len(y) = {} is not long "
        #         "enough for using max_T = {} " "samples for estimation and "
        #         "the remaining for the " "out of sample error calculation. "
        #         "Ensure that we have " "max_T <= len(y) - 100."
        #         "".format(len(y), max_T))

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


def compute_gc_score(xi_i, xi_ij, T, p_lags, F_distr=False):
    with np.errstate(divide="ignore", invalid="ignore"):
        if F_distr:
            F = (T - p_lags) * ((xi_i[:, None] / xi_ij) - 1) / p_lags
        else:
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
def fast_compute_xi(R, T, n_min=0, n_max=np.inf):
    """
    Calculates Xi[n_min: n_max, :]
    """
    _, n, _ = R.shape
    n_min = max(n_min, 0)
    n_max = min(n_max, n)

    xi_i = np.nan * np.zeros(n)
    xi_ij, p_ij = np.nan * np.zeros((n, n)), np.nan * np.zeros((n, n))

    for i in range(n_min, n_max):
        xi_i[i], p_i = _fast_univariate_AR_error(R[:, i, i], T)
        xi_ij[i, i] = xi_i[i]
        p_ij[i, i] = p_i

    # Recall that we are computing a horizontal band
    # as well as the transposed vertical band

    for i in range(n_min, n_max):
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
def compute_pairwise_gc(X, max_lag=10, F_distr=False):
    T, _, _ = *X.shape, max_lag
    xi_i, xi_ij, p_i, p_ij = compute_xi(
        X, max_lag)
    F = compute_gc_score(xi_i, xi_ij, T, p_ij, F_distr=F_distr)
    return F, p_ij  # p_i is actually irrelevant.


def normalize_gc_score(F, p, T=None, F_distr=False):
    if F_distr:
        if T is None:
            raise ValueError("If F_distr=True, we also require T.")
        F = sps.f.cdf(F, p, T - p)
    else:
        F = sps.chi2.cdf(F, p)
    F[np.isnan(F)] = 0.0
    return F


def fast_compute_pairwise_gc(X, max_lag=10, F_distr=False, k_cores=1):
    """
    This is /dramatically/ faster than compute_pairwise_gc.

    TODO: However there are significant discrepancies!
    TODO: There is something clearly wrong about this implementation.
    """
    T, n, = X.shape

    R = compute_covariances(X, max_lag)
    if k_cores == 1:
        xi_i, xi_ij, p_ij = fast_compute_xi(R, T, n_min=0, n_max=n)
    elif k_cores > 1:
        xi_i, xi_ij, p_ij = _par_fast_compute_pairwise_gc(
            n, T, R, k_cores=k_cores)
    else:
        raise AssertionError("Must have k_cores >= 1!")
    return compute_gc_score(xi_i, xi_ij, T, p_ij, F_distr=F_distr), p_ij


def _par_fast_compute_pairwise_gc(n, T, R, k_cores=2):
    xi_i = np.nan * np.ones(n)
    xi_ij = np.nan * np.ones((n, n))
    p_ij = np.nan * np.ones((n, n))

    pool = Pool(k_cores)

    # Heuristic for load balancing
    n_split = list(map(int,
                       n * (1 - (np.linspace(0, 1, k_cores + 1) ** 2)[::-1])))
    zip_n = list(zip(n_split[:-1], n_split[1:]))

    args = [(R, T, n_min, n_max) for n_min, n_max in zip_n]
    res = pool.starmap(fast_compute_xi, args)
    pool.close()

    for i, n_min_n_max in zip(range(len(zip_n)), zip_n):
        n_min, n_max = n_min_n_max
        _xi_i, _xi_ij, _p_ij = res[i]
        xi_i[n_min:n_max] = _xi_i[n_min: n_max]
        xi_ij[n_min:n_max, :n_max] = _xi_ij[n_min:n_max, :n_max]
        xi_ij[:n_max, n_min:n_max] = _xi_ij[:n_max, n_min:n_max]
        p_ij[n_min:n_max, :n_max] = _p_ij[n_min:n_max, :n_max]
        p_ij[:n_max, n_min:n_max] = _p_ij[:n_max, n_min:n_max]
    pool.join()
    return  xi_i, xi_ij, p_ij


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
    # TODO: Replace this with ts_lasso

    _, n = X.shape
    G = make_complete_digraph(n)
    G = attach_X(G, X)

    estimate_B(G, max_lag, copy_G=False, max_T=max_T, method=method)
    G = remove_zero_filters(G, "b_hat(z)", copy_G=False)
    return G


def alasso_fista_estimate_dense_graph(X, max_lag=10, max_T=None, nu=1.25,
                                      eps=1e-3, full_path=False):
    T, n = X.shape
    if max_T is None:
        max_T = T

    B, _, _, _ = fit_VAR(X[:max_T, :], max_lag, nu=nu, eps=eps,
                         full_path=full_path)

    G = make_complete_digraph(n)
    G = attach_X(G, X)

    for i in range(n):
        for j in range(n):
            b_hatz = B[:, i, j]
            G[j][i]["b_hat(z)"] = b_hatz

    if max_T < T:
        p = len(B)
        z0 = np.hstack([X[max_T - tau] for tau in range(1, p + 1)])
        var_sys = gcg_to_var(G, filter_attr="b_hat(z)", assert_stable=False)
        var_sys.reset(x_0=z0)
        R = var_sys.compute_residuals(X[max_T:])
        for i in G.nodes:
            r = R[:, i]
            G.nodes[i]["r"] = r
            G.nodes[i]["sv^2_hat"] = np.var(r)

    G = remove_zero_filters(G, "b_hat(z)", copy_G=False)
    return G


def pwgc_estimate_graph(X, max_lags=10, alpha=0.05,
                        method="lstsqr"):
    T, n = X.shape

    F, P = fast_compute_pairwise_gc(X, max_lag=max_lags,
                                    k_cores=int(1 + np.log(n)))

    # Screen edges via benjamini hochberg criterion
    P_edges = normalize_gc_score(F, P, T, F_distr=True)  # p-values are just 1 - F
    P_values = 1 - P_edges[~np.eye(n, dtype=bool)].ravel()
    t_bh = benjamini_hochberg(P_values, alpha=alpha, independent=False)

    # Estimate a strongly causal graph
    G_hat = pw_scg(F, P_edges, t_bh)
    G_hat = attach_X(G_hat, X, prop_name="x")
    G_hat = add_self_loops(G_hat, copy_G=False)

    G_hat = estimate_B(G_hat, max_lags, copy_G=False,
                       method=method)
    G_hat = remove_zero_filters(G_hat, "b_hat(z)", copy_G=False)
    return G_hat


def estimate_graph(X, G, max_lags=10, method="lasso", alpha=0.05,
                   fast_mode=True, F_distr=False):
    """
    Produce an estimated graph from the data X.

    This function suffers from poor design as I am also needing to
    pass in a "true" graph G in order to attach the right data to
    G_hat and calculate the estimate of sigma_v^2.

    TODO: This probably needs to be fixed before doing any application
    """
    T, n = X.shape

    # Compute the pairwise errors and filter sizes
    if fast_mode:
        F, P = fast_compute_pairwise_gc(X, max_lag=max_lags, F_distr=F_distr,
                                        k_cores=int(1 + np.log(n)))
    else:
        F, P = compute_pairwise_gc(X, max_lag=max_lags, F_distr=F_distr)

    # Screen edges via benjamini hochberg criterion
    P_edges = normalize_gc_score(F, P, T, F_distr=F_distr)  # p-values are just 1 - F
    P_values = 1 - P_edges[~np.eye(n, dtype=bool)].ravel()
    t_bh = benjamini_hochberg(P_values, alpha=alpha, independent=False)

    # Estimate a strongly causal graph
    G_hat = pw_scg(F, P_edges, t_bh)
    G_hat = attach_node_prop(G_hat, G, prop_attach="x", prop_from="x")
    G_hat = add_self_loops(G_hat, copy_G=False)

    # Restimate the coefficients
    if method in {"lasso", "glasso", "alasso"}:
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
    mcc = _compute_mcc(TP, TN, FP, FN)
    return mcc


def compute_fdr_score(N_edges, N_hat_edges, N_intersect_edges, n_nodes):
    TP, TN, FP, FN = compute_tp_tn(N_edges, N_hat_edges, N_intersect_edges,
                                   n_nodes)
    return FP / (TP + FP)


def _compute_mcc(TP, TN, FP, FN):
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) *
                                        (TN + FP) * (TN + FN))
    return mcc


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
