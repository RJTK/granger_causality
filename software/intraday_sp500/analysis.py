import numpy as np
import pandas as pd
import seaborn as sns

import networkx as nx

from scipy.stats import boxcox
from matplotlib import pyplot as plt

from pwgc import pwgc_estimate_graph


def main():
    # Read in some data
    D = pd.read_csv("dataset.csv", nrows=1000, index_col=0,
                    header=[0, 1], parse_dates=False)

    # Look at closing prices only
    D_close = D.loc[:, (slice(None), "close")]
    D_close.columns = D_close.columns.droplevel(1)
    sp500 = list(D_close.columns)

    # And also keep the volume
    D_vol = D.loc[:, (slice(None), "volume")]
    D_vol.columns = D_vol.columns.droplevel(1)

    # Plot the NaNs
    # plot_nans(D_close)
    # plot_nans(D_vol)

    # Fill the nans in...
    D_close = fill_nans(D_close)
    D_vol = fill_nans(D_vol)

    # Apply some simple transforms (i.e. diffing)
    D_close, D_vol = price_vol_transform(D_close, D_vol)

    fig, ax = plt.subplots(1, 1)
    ax_vol = ax.twinx()
    ax_vol.plot(D_vol.loc[:, "AAPL"].to_numpy(), linewidth=2, color="r", label="volume")
    ax.plot(D_close.loc[:, "AAPL"].to_numpy(), linewidth=2, color="b",
            label="price diff")
    ax_vol.legend()
    ax_vol.grid(False)
    plt.show()

    X = D_close.to_numpy()

    graph_estimates = []
    max_lag = 2
    G_e = pwgc_estimate_graph(X, max_lags=max_lag, method="alasso")
    graph_estimates.append(G_e)
    for i in range(3):
        G_r = get_residual_graph(graph_estimates[-1])
        X_r = get_X(G_r)
        G_e = pwgc_estimate_graph(X_r, max_lags=max_lag, method="alasso",
                                  alpha=0.01)
        graph_estimates.append(G_e)

    G_sp500_hat = combine_graphs(graph_estimates)

    G_sp500_hat = attach_X(G_sp500_hat, X)

    T_max = 750
    G_sp500_hat = estimate_B(G_sp500_hat, max_lag=max_lag, method="lstsqr",
                             max_T=T_max)

    X_r = get_X(G_sp500_hat, prop="r")

    G_sp500_hat = nx.relabel_nodes(G_sp500_hat,
                                   {i: sp500[i] for i in range(len(sp500))})


    # draw_graph(G_sp500_hat)
    nx.readwrite.gexf.write_gexf(G_sp500_hat, path="./G_sp500_hat.gexf")

    i = 10
    plt.plot(X[6:, i], color="b", alpha=0.75, linewidth=2)
    plt.plot(X_r[:, i], color="r", alpha=0.75)
    plt.show()

    plt.plot(X[6:, i] - X_r[:, i])
    plt.show()

    i = 12
    plt.plot(X[-247:, i], color="b", alpha=0.75, linewidth=2)
    plt.plot(X_r[:, i], color="r", alpha=0.75, linewidth=1)
    plt.show()

    V = [np.var(X[-247:, i]) - np.var(X_r[:, i]) for i in range(500)]

    deg_seq = [d for n, d in G_sp500_hat.degree()]

    page_rank = sorted(list(nx.pagerank(G_sp500_hat).items()),
                            key=lambda i: i[1], reverse=True)

    deg_centrality = sorted(list(nx.degree_centrality(G_sp500_hat).items()),
                                 key=lambda i: i[1], reverse=True)
    return


def plot_nans(D):
    # Check out the NaN
    nan_prop = D.isna()\
        .sum()\
        .sort_values() / len(D)
    nan_prop.plot(linewidth=2)
    plt.show()
    return


def fill_nans(D):
    D = D.interpolate(method="linear")
    D = D.ffill().bfill()
    return D


def price_vol_transform(D_close, D_vol):
    D_close = D_close.diff()[1:]
    D_vol = D_vol[:-1]
    return D_close, D_vol
