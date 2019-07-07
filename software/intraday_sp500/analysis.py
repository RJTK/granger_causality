import numpy as np
import pandas as pd
import seaborn as sns
import pandas_market_calendars as pmc
import networkx as nx

from datetime import datetime
from scipy.stats import boxcox
from matplotlib import pyplot as plt

from pwgc.gc_methods import (pwgc_estimate_graph, get_residual_graph,
                             get_X, combine_graphs, estimate_B,
                             attach_X)

T_in_day = 391


def main():
    # Read in some data
    D = pd.read_csv("dataset.csv", nrows=10000, index_col=0,
                    header=[0, 1], parse_dates=False)

    # Deal with calendars
    start_date = D.index[0][:10]
    end_date = D.index[-1][:10]
    nyse_cal = pmc.get_calendar("NYSE")
    nyse_times = nyse_cal.schedule(start_date=start_date,
                                   end_date=end_date)

    # Pick out the days of the week
    D["weekday"] = list(map(lambda dt:
                            datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").weekday(),
                            D.index))

    def min_since_930(dt):
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        dt0 = dt.replace(hour=9, minute=30)
        delta = dt - dt0
        return delta.seconds // 60

    D["min"] = list(map(min_since_930, D.index))

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
    cut = 100
    D_close = D_close[np.logical_and(D["min"] > cut, D["min"] < T_in_day - cut)]

    # fig, ax = plt.subplots(1, 1)
    # ax_vol = ax.twinx()
    # ax_vol.plot(D_vol.loc[:, "AAPL"], linewidth=2, color="r", label="volume")
    # ax.plot(D_close.loc[:, "AAPL"], linewidth=2, color="b",
    #         label="price diff")
    # ax.plot(2 + D_close.loc[np.logical_and(D["min"] > 20, D["min"] < 370), "AAPL"],
    #         color="g")
    # ax_vol.legend()
    # ax_vol.grid(False)
    # ax.set_xticks([])
    # plt.show()

    X = D_close.to_numpy()
    T, n = X.shape
    days_per_iter = 8
    max_lag = 2
    cut_T_in_day = T_in_day - 2 * cut
    max_T = (days_per_iter - 1) * cut_T_in_day

    all_V = []
    for it in range(T // cut_T_in_day - days_per_iter + 1):
        print(it)
        X_day = X[it * cut_T_in_day: (it + days_per_iter) * cut_T_in_day]
        V = full_graph_estimates(X_day, max_T, max_lag,
                                 N_iters=1)
        plt.plot(np.sort(V), color="b", linewidth=2, alpha=0.5)
        all_V.append(V)
    plt.show()

    V = np.vstack(all_V)
    v = V.ravel()
    v = v[v != 0]
    sps.probplot(v, plot=plt)
    plt.show()

    print(np.sum(v))
    print(np.mean(v))
    return


def full_graph_estimates(X, max_T, max_lag, N_iters=4):
    graph_estimates = []
    G_e = pwgc_estimate_graph(X[:max_T], max_lags=max_lag, method="lstsqr")
    graph_estimates.append(G_e)
    for i in range(N_iters - 1):
        G_r = get_residual_graph(graph_estimates[-1])
        X_r = get_X(G_r)
        G_e = pwgc_estimate_graph(X_r, max_lags=max_lag, method="lstsqr",
                                  alpha=0.01)
        graph_estimates.append(G_e)

    G_sp500_hat = combine_graphs(graph_estimates)
    G_sp500_hat = attach_X(G_sp500_hat, X)
    G_sp500_hat = estimate_B(G_sp500_hat, max_lag=max_lag, method="lstsqr",
                             max_T=max_T)
    X_r = get_X(G_sp500_hat, prop="r")
    X_original = X[max_T + max_lag:, :]

    # i = 0
    # x_real = np.cumsum(X_original, axis=0)
    # x_hat = (x_real + X_pred)[:-1]
    # plt.plot(x_hat[:, i], label="pred")
    # plt.plot(x_real[1:, i], label="actual")
    # plt.plot(x_real[:-1, i], label="baseline")
    # plt.legend()
    # plt.show()

    # i = 0
    # plt.plot(X_r[:, i])
    # plt.plot(X_original[:, i])
    # plt.show()

    # The variance of the signal itself in comparison to the variance
    # if we subtract our predictions.  These numbers should be positive
    # if our prediction was useful, and negative otherwise.
    V = np.var(X_original, axis=0) - np.var(X_r, axis=0)
    return V

    # G_sp500_hat = nx.relabel_nodes(G_sp500_hat,
    #                                {i: sp500[i] for i in range(len(sp500))})


    # draw_graph(G_sp500_hat)
    # nx.readwrite.gexf.write_gexf(G_sp500_hat, path="./G_sp500_hat.gexf")

    # i = 10
    # plt.plot(X[6:, i], color="b", alpha=0.75, linewidth=2)
    # plt.plot(X_r[:, i], color="r", alpha=0.75)
    # plt.show()

    # plt.plot(X[6:, i] - X_r[:, i])
    # plt.show()

    # i = 12
    # plt.plot(X[-247:, i], color="b", alpha=0.75, linewidth=2)
    # plt.plot(X_r[:, i], color="r", alpha=0.75, linewidth=1)
    # plt.show()

    # deg_seq = [d for n, d in G_sp500_hat.degree()]

    # page_rank = sorted(list(nx.pagerank(G_sp500_hat).items()),
    #                         key=lambda i: i[1], reverse=True)

    # deg_centrality = sorted(list(nx.degree_centrality(G_sp500_hat).items()),
    #                              key=lambda i: i[1], reverse=True)
    # return


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
