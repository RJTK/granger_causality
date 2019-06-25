"""
Some simple checks that my fitting implementations are actually correct.
"""

import numpy as np
import matplotlib.pyplot as plt

from pwgc.gc_methods import (estimate_b_lstsqr,
                             estimate_b_lasso,
                             estimate_b_alasso,
                             _compute_mcc)


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
