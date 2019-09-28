import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import solve_discrete_lyapunov


def plot_vars(x, title=""):
    plt.plot(x[:, 0], linewidth=2, alpha=0.75, label="$x_0$")
    plt.plot(x[:, 1], linewidth=2, alpha=0.75, label="$x_1$")
    plt.plot(x[:, 2], linewidth=2, alpha=0.75, label="$x_2$")
    plt.xlabel("$t$")
    plt.title(title)
    plt.legend()
    return


def var1_example():
    n, T = 3, 1000
    t = np.linspace(0, 1, T)
    Sv = np.diag(np.array([2.0, 1.0, 0.5]))
    B = np.zeros((3, 3))

    B[0, 0] = 0.5
    B[1, 1] = 0.7
    B[2, 2] = 0.99

    x = np.zeros((T, n))
    v = np.random.multivariate_normal(mean=np.zeros(n),
                                      cov=Sv, size=T)

    Sx = solve_discrete_lyapunov(a=B.T, q=Sv)

    x[0, :] = np.random.multivariate_normal(mean=np.zeros(n),
                                            cov=Sx)
    for t in range(1, T):
        x[t, :] = B @ x[t - 1, :] + v[t]

    plot_vars(x, "Independent Processes")
    plt.savefig("figures/var1_example.png")
    plt.savefig("figures/var1_example.pdf")
    plt.show()
    return


def var2_example():
    n, T = 3, 1000
    t = np.linspace(0, 1, T)
    Sv = np.diag(np.array([2.0, 1.0, 0.5]))
    B = np.zeros((3, 3))

    B[0, 0] = 0.5
    B[1, 1] = 0.7
    B[2, 2] = 0.99

    B_lag = np.zeros_like(B)
    B_lag[0, 2] = 0.8
    B_lag[1, 2] = 0.7

    x = np.zeros((T, n))
    v = np.random.multivariate_normal(mean=np.zeros(n),
                                      cov=Sv, size=T)

    Sx = solve_discrete_lyapunov(a=B.T, q=Sv)
    t0 = 150
    x[0, :] = np.random.multivariate_normal(mean=np.zeros(n),
                                            cov=Sx)
    for t in range(1, t0):
        x[t, :] = B @ x[t - 1, :] + v[t]
    for t in range(t0, T):
        x[t, :] = B @ x[t - 1, :] + B_lag @ x[t - t0, :] + v[t]

    plot_vars(x, "$x_2$ driving $x_0, x_1$")
    plt.savefig("figures/var2_example.png")
    plt.savefig("figures/var2_example.pdf")
    plt.show()
    return


if __name__ == "__main__":
    var1_example()
    var2_example()
