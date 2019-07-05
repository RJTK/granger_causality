import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer

COLOR1 = "#377eb8"
COLOR2 = "#e41a1c"
COLOR3 = "#4daf4a"
COLOR4 = "#984ea3"
COLOR5 = "#e41a1c"


def plotting_model(x, y):
    """
    Fit a simple Kernel Ridge Regression model, purely for plotting
    purposes.
    """
    tx = PowerTransformer(method="box-cox", standardize=True)
    X = x[:, None]
    X = tx.fit_transform(X)

    kernel = RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)

    gpr.fit(X, y)
    y_hat = gpr.predict(X)
    return y_hat
