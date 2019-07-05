import unittest
import numpy as np
import matplotlib.pyplot as plt

from pwgc.gc_methods import (_fast_univariate_AR_error,
                             _fast_bivariate_AR_error,
                             compute_covariances,
                             form_bivariate_covariance,
                             fast_compute_xi, fast_compute_pairwise_gc)
from pwgc.var_system import (random_tree_dag, get_X, drive_gcg)


class TestARFitting(unittest.TestCase):
    def SetUp(self):
        np.random.seed(0)
        return

    def _create_data(self):
        n_nodes, p_lags, max_lags = 30, 5, 10

        T_max = 2000

        random_graph = lambda: random_tree_dag(
            n_nodes, p_lags, pole_rad=0.75)
        sv2_true = 0.5 + np.random.exponential(0.5, size=n_nodes)
        G = random_graph()
        drive_gcg(G, T_max, sv2_true, filter_attr="b(z)")
        X = get_X(G)
        X = X - np.mean(X, axis=0)[None, :]
        return X

    def test_univariate_AR_error001(self):
        # This only really checks that it runs without an error
        X = self._create_data()
        p_max = 10
        T, n = X.shape

        R = compute_covariances(X, p=p_max)
        for i in range(n):
            r = R[:, i, i]
            err, p_opt = _fast_univariate_AR_error(r, T)
        return

    def test_univariate_AR_error002(self):
        # This only really checks that it runs without an error
        X = self._create_data()
        p_max = 15
        T, n = X.shape

        Eps = np.zeros((p_max + 1, n))  # From levinson
        SV2 = np.zeros((p_max + 1, n))  # Direct lstsqr

        R = compute_covariances(X, p=p_max)
        for i in range(n):
            r = R[:, i, i]
            _, _, eps = lev_durb(r)
            Eps[:, i] = eps
            bic = compute_bic(eps, T, s=1)
            p_opt = np.argmax(bic)
            plt.plot(bic, color="b", alpha=0.5)
            plt.scatter(p_opt, bic[p_opt], color="r", marker="o")

        for i in range(n):
            bic = np.zeros(p_max + 1)
            for p in range(p_max + 1):
                if p == 0:
                    bic[0] = -np.log(np.var(X[:, i]))
                    SV2[0, i] = np.var(X[:, i])
                else:
                    X_, y_ = form_Xy(X_raw=X[:, i][:, None],
                                     y_raw=X[:, i], p=p)
                    sv2_p = _compute_AR_error(X_, y_)
                    bic_p = -np.log(sv2_p) - p * np.log(T) / T
                    bic[p] = bic_p
                    SV2[p, i] = sv2_p
            p_opt = np.argmax(bic)
            plt.plot(bic, color="g", alpha=0.5)
            plt.scatter(p_opt, bic[p_opt], color="m", marker="o")
        plt.show()

        plt.imshow(Eps - SV2); plt.show()
        return

    def test_bivariate_AR_error001(self):
        # This only really checks that it runs without an error
        X = self._create_data()
        p_max = 15
        T, n = X.shape

        R = compute_covariances(X, p=p_max)
        for i in range(n):
            for j in range(i):
                Rij = form_bivariate_covariance(R, i, j)
                err_i, err_j, p_opt =\
                    _fast_bivariate_AR_error(Rij, T)

                Rji = form_bivariate_covariance(R, j, i)
                err_jT, err_iT, p_optT =\
                    _fast_bivariate_AR_error(Rji, T)
                np.testing.assert_almost_equal(err_i, err_iT)
                np.testing.assert_almost_equal(err_j, err_jT)
        return

    def test_bivariate_AR_error002(self):
        # This only really checks that it runs without an error
        X = self._create_data()
        p_max = 15
        T, n = X.shape

        Eps = np.zeros((p_max + 1, n, n))  # From levinson
        SV2 = np.zeros((p_max + 1, n, n))  # Direct lstsqr

        R = compute_covariances(X, p=p_max)
        for i in range(n):
            for j in range(i):
                Rij = form_bivariate_covariance(R, i, j)
                _, _, S = whittle_lev_durb(Rij)
                Eps[:, i, j] = np.linalg.det(S)
                bic = compute_bic(Eps[:, i, j], T, s=4)
                p_opt = np.argmax(bic)
                plt.plot(bic, color="b", alpha=0.5, linewidth=0.5)
                plt.scatter(p_opt, bic[p_opt], color="r", marker="o")

        for i in range(n):
            for j in range(i):
                bic = np.zeros(p_max + 1)
                for p in range(p_max + 1):
                    Xij = X[:, [i, j]]
                    if p == 0:
                        bic[0] = -np.log(np.var(Xij))
                        SV2[0, i] = np.var(Xij)
                    else:
                        X_, y_ = form_Xy(
                            X_raw=np.hstack((Xij[:, 0][:, None],
                                             Xij[:, 1][:, None])),
                                            y_raw=Xij[:, 0], p=p)
                        sv2_p = _compute_AR_error(X_, y_)
                        bic_p = -np.log(sv2_p) - 4 * p * np.log(T) / T
                        bic[p] = bic_p
                        SV2[p, i] = sv2_p
                p_opt = np.argmax(bic)
                plt.plot(bic, color="g", alpha=0.5)
                plt.scatter(p_opt, bic[p_opt], color="m", marker="o")
        plt.show()

        # Eps = Eps.reshape(p_max + 1, 30**2)
        # SV2 = SV2.reshape(p_max + 1, 30**2)
        # plt.imshow(Eps - SV2, aspect="auto"); plt.show()

        # Err = []
        err = np.mean(np.log(np.abs(Eps[:, np.array(np.tri(n, n, -1), dtype=bool)] - SV2[:, np.array(np.tri(n, n, -1), dtype=bool)])))
        Err.append(err)
        return

    def test_compute_xi(self):
        X = self._create_data()
        p_max = 15

        fast_compute_xi(X, p_max)
        return


    def test_pw_gc(self):
        X = self._create_data()
        p_max = 15

        F_fast, P_fast = fast_compute_pairwise_gc(X, p_max)
        F_slow, P_slow = compute_pairwise_gc(X, p_max)

        G_fast = normalize_gc_score(F_fast, P_fast)
        G_slow = normalize_gc_score(F_slow, P_slow)

        plt.scatter(G_fast.ravel(), G_slow.ravel())
        plt.plot([0, 1], [0, 1])
        plt.show()

        plt.scatter(np.log(F_fast).ravel(),
                    np.log(F_slow).ravel())
        plt.plot([0, 10], [0, 10])
        plt.show()
        return
