import unittest
import numpy as np
import matplotlib.pyplot as plt

from pwgc.gc_methods import (_fast_univariate_AR_error,
                             _fast_bivariate_AR_error,
                             compute_covariances,
                             form_bivariate_covariance,
                             fast_compute_xi, fast_compute_pairwise_gc,
                             estimate_b_lstsqr,
                             estimate_b_lasso,
                             estimate_b_alasso,
                             _compute_mcc)
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


class TrackErrors:
    def __init__(self, N_iters, p, p0):
        self.N_iters = N_iters
        self.p = p
        self.p0 = p0
        self.B_err = np.empty(N_iters)
        self.l2_loss = np.empty(N_iters)
        self.true_pos = np.empty(N_iters)  # True positives
        self.true_neg = np.empty(N_iters)  # True negatives
        self.pos = np.empty(N_iters)  # True positives
        self.neg = np.empty(N_iters)  # True negatives
        return

    def update(self, n_it, b, b_true, y, y_hat):
        self.B_err[n_it] = np.linalg.norm(b - b_true)**2 / self.p
        self.l2_loss[n_it] = np.linalg.norm(y - y_hat)**2 / len(y)
        self.true_pos[n_it] = np.sum(np.logical_and(b != 0,
                                                    b_true != 0))
        self.pos[n_it] = np.sum(b != 0)

        self.true_neg[n_it] = np.sum(np.logical_and(np.abs(b) == 0,
                                                    np.abs(b_true) == 0))
        self.neg[n_it] = np.sum(b == 0)
        return

    def get_results(self):
        fp = self.pos - self.true_pos
        fn = self.neg - self.true_neg
        mcc = _compute_mcc(self.true_pos, self.true_neg,
                           fp, fn)
        return self.B_err, self.l2_loss, mcc


def example_linear_regression():
    np.random.seed(0)
    p = 20
    p0 = 8

    sigma = 2.5
    N_iters = 250

    errs_lstsqr, errs_lasso, errs_alasso = [TrackErrors(N_iters, p, p0)
                                            for _ in range(3)]

    sample_points = np.array(list(map(int, np.linspace(10, 5e5, N_iters))))
    n_test = 10000
    for i, n in enumerate(sample_points):
        print("i = {} / {}, n = {}\r".format(i + 1, N_iters, n))
        noise = sigma * np.random.normal(size=n + n_test)
        b_true = np.random.laplace(size=p)
        b_true[p0:] = 0.0
        # L = np.random.standard_t(size=(p, p), df=3)
        L = np.random.normal(size=(p, p))  # Make X dependent
        X = np.random.normal(size=(n + n_test, p))
        X[2:] += 0.5 * np.random.normal() * X[:-2]
        X = X @ L

        X = X / np.std(X)
        y = X @ b_true + noise

        # TODO: WTF?  lmbda appears to do nothing.
        b_lstsqr = estimate_b_lstsqr(X[:n], y[:n], lmbda=np.inf)
        b_lasso = estimate_b_lasso(X[:n], y[:n])
        b_alasso = estimate_b_alasso(X[:n], y[:n], nu=0.5,
                                     lmbda_lstsqr=2.0)

        y_lstsqr = X[n:] @ b_lstsqr
        y_lasso = X[n:] @ b_lasso
        y_alasso = X[n:] @ b_alasso

        errs_lstsqr.update(i, b_lstsqr, b_true, y[n:], y_lstsqr)
        errs_lasso.update(i, b_lasso, b_true, y[n:], y_lasso)
        errs_alasso.update(i, b_alasso, b_true, y[n:], y_alasso)

    b_err_lstsqr, l2_loss_lstsqr, mcc_lstsqr = errs_lstsqr.get_results()
    b_err_lasso, l2_loss_lasso, mcc_lasso = errs_lasso.get_results()
    b_err_alasso, l2_loss_alasso, mcc_alasso = errs_alasso.get_results()

    mcc_lasso[np.isnan(mcc_lasso)] = 0.0
    mcc_alasso[np.isnan(mcc_alasso)] = 0.0

    l2_loss_lstsqr = np.log(l2_loss_lstsqr / sigma**2)
    l2_loss_lasso = np.log(l2_loss_lasso / sigma**2)
    l2_loss_alasso = np.log(l2_loss_alasso / sigma**2)

    b_err_lstsqr = np.log(b_err_lstsqr)
    b_err_lasso = np.log(b_err_lasso)
    b_err_alasso = np.log(b_err_alasso)

    fig, axes = plt.subplots(3, 1, sharex=True)
    plot_results(sample_points, b_err_lstsqr, b_err_lasso, b_err_alasso,
                 axes[0], title="b_err", ylabel="b_err")

    plot_results(sample_points, l2_loss_lstsqr, l2_loss_lasso,
                 l2_loss_alasso,
                 axes[1], title="log relative l2_loss",
                 ylabel="log relative l2 loss",
                 legend=False)

    plot_results(sample_points, mcc_lstsqr, mcc_lasso, mcc_alasso,
                 axes[2], title="mcc", ylabel="mcc", xlabel="n_samples",
                 legend=False)

    fig.suptitle("Estimation accuracy over number of samples")
    plt.show()
    return


def plot_results(x, lstsqr, lasso, alasso, ax,
                 title="", ylabel="", xlabel="", legend=True):
    ax.scatter(x, lstsqr, color="r", label="lstsqr", marker="o",
               alpha=0.65)
    ax.scatter(x, lasso, color="b", label="lasso", marker="^",
               alpha=0.65)
    ax.scatter(x, alasso, color="g", label="alasso", marker="s",
               alpha=0.65)

    try:
        ax.plot(x, fit_curve(x, lstsqr), color="r", linewidth=2)
    except ValueError:
        pass

    ax.plot(x, fit_curve(x, lasso), color="b", linewidth=2)
    ax.plot(x, fit_curve(x, alasso), color="g", linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()
    return ax


def fit_curve(x, y):
    X = np.vstack((x, x**2, np.sqrt(x), np.ones_like(x))).T
    b_hat = estimate_b_lstsqr(X, y)
    return X @ b_hat


if __name__ == "__main__":
    example_linear_regression()
