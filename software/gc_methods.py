import networkx as nx
import numpy as np

from sklearn.linear_model import LinearRegression, LassoLarsIC


def estimate_b_lstsqr(y, X):
    """
    Simple direct estimate of a linear filter y_hat = Xb.
    """
    lr = LinearRegression(fit_intercept=False, normalize=False,
                          copy_X=True)
    lr.fit(X, y)
    return lr.coef_


def estimate_b_lasso(y, X, criterion="bic"):
    """
    Estimate y_hat = Xb via LASSO and using BIC to choose regularizers.
    """
    lasso = LassoLarsIC(criterion=criterion, fit_intercept=False,
                        normalize=False)
    lr = LinearRegression(fit_intercept=False, normalize=False,
                          copy_X=False)

    lasso.fit(X, y)
    w = lasso.coef_

    # Use lasso only to select the support
    X_lasso = X[:, [i for i, wi in enumerate(w) if wi != 0]]
    w_lr = lr.fit(X_lasso, y).coef_
    w[[i for i, wi in enumerate(w) if wi != 0]] = w_lr
    return w


def estimate_B(X, G, max_lag=10):
    """
    Estimate x(t) = B(z)x(t) respecting the topology of G.
    """
    
    return


def example_graph():
    N = list(range(1, 7))
    E = [(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]

    G = nx.DiGraph()
    G.add_nodes_from(N)
    G.add_edges_from(E)  # This will add nodes if they aren't already there.

    # set(G.predecessors(i)): set of parents of i in G
    # set(nx.ancestors(G, i)): set of ancestors of i in G
    return



def example():

    def sample_x():
        T = 2000
        th = 2.25 * np.pi
        a = 2 * np.cos(th)
        e = np.random.normal(size=T)
        x = np.zeros(T + 2)
        for t in range(T):
            x[t] = a * x[t - 1] - x[t - 2] + e[t]
        return x

    [plt.plot(sample_x(), linewidth=.75, alpha=0.8) for _ in range(20)]
    plt.show()
    return
