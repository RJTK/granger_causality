import seaborn.apionly as sns
import statsmodels as sm
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scipy.stats as sps
import xarray as xr
import pandas_market_calendars as mcal

from itertools import chain
from statsmodels.tsa.stattools import (adfuller, acf, pacf,
                                       grangercausalitytests)

DATA_DIR = "data/"

def get_stock_name(f):
    return f.split(".")[0]


def read_stock_csv(directory, file_name, cols=None):
    """
    Intended to read files formatted as "<ticker>.us.txt" with data
    (Date, Open, High, Low, Volume, OpenInt)
    
    """
    if cols is None:
        cols = ["Close", "Volume"]
    if "Date" not in cols:
        cols = cols + ["Date"]

    name = get_stock_name(file_name)
    try:
        # index_col then reset_index is to ensure date types are parsed
        D = pd.read_csv(directory + file_name, parse_dates=True,
                        infer_datetime_format=True,
                        usecols=cols, dtype={"Volume": float},
                        index_col="Date").reset_index()
        D["Symbol"] = name
    except pd.errors.EmptyDataError:
        return None
    else:
        return D


def read_sent_csv(file_path):
    """
    Reads the sentiment analysis data
    """
    sent = pd.read_csv(file_path, index_col=0,
                       infer_datetime_format=True,
                       parse_dates=["date"])\
        .rename({"symbol": "Symbol",
                 "sentiment_signal": "sent",
                 "date": "Date"},
                axis="columns")
    sent["Symbol"] = sent["Symbol"].str.lower()
    sent = sent.set_index(["Symbol", "Date"])\
        .sort_index()
    return sent


def stock_filter(D, min_average_volume=50000,
                 min_date=None, max_date=None,
                 inclusive_dates=False):
    """
    Filter out stocks on dates and volumes.
    """
    def conv_date(d):
        if isinstance(d, str):
            return pd.Timestamp(d)
        else:
            return d

    if D is None:
        return None

    min_date, max_date = map(conv_date, [min_date, max_date])

    try:
        if D["Volume"].mean() < min_average_volume:
            return None
    except KeyError:
        return None


    if inclusive_dates:
        if D.loc[D.index[-1], "Date"] < max_date:
            return None
        if D.loc[D.index[0], "Date"] > min_date:
            return None

    if min_date is not None:
        D = D.loc[D["Date"] >= min_date]
    if max_date is not None:
        D = D.loc[D["Date"] <= max_date]
    if len(D) == 0:
        return None

    return D


def load_stock_data(ret_symbols=False, ret_sectors=False,
                    min_date=None, max_date=None):
    """
    Loads stock data for S&P500.
    """
    directory = DATA_DIR + "raw/Stocks/"
    stock_files = [f for f in os.listdir(directory)
                   if f.endswith("txt")]
    sp500 = pd.read_csv("sp500.txt")
    symbols = set(sp500["Symbol"].map(lambda s: s.lower()))

    def read_filter(f):
        if get_stock_name(f) not in symbols:
            return None
        else:
            D = read_stock_csv(directory, f)

        if (min_date is not None) and (max_date is not None):
            D = stock_filter(D, min_date=min_date, max_date=max_date,
                             inclusive_dates=True)
        else:
            D = stock_filter(D)
        return D

    D = pd.concat([D for D in map(read_filter, stock_files)
                   if D is not None])\
                   .set_index(["Symbol", "Date"])
    ret = [D]

    # Pull out industries
    symbols = D.index.levels[0].to_numpy()
    if ret_symbols:
        ret = ret + [symbols]

    if ret_sectors:
        sp500["Symbol"] = sp500["Symbol"].str.lower()
        sectors = sp500.set_index("Symbol")\
            .loc[symbols]\
            .reset_index()\
            .groupby("Sector")\
            .apply(lambda s: s["Symbol"])
        ret = ret + [sectors]
    return ret


def load_sent_data():
    file_path = "data/raw/sentiment_data.csv"
    sent = read_sent_csv(file_path)
    return sent


def repair_data(D):
    """
    Does a bunch of fillins of sentiment data:

    0. remove symbols with no sentiment
    1. fill forward
    2. cut off first 50 days
    3. fill each symbol with their own mean  (this is non-causal but few filled)
    """
    no_sent_data = D.isna()\
        .groupby(level=0)\
        .all()\
        .any(axis=1)
    has_sent_data = no_sent_data[~no_sent_data].index
    D = D.loc[has_sent_data]

    D = D.groupby(level=0).fillna(method="ffill")
    D = D.loc[(slice(None), D.index.levels[1][50:]), :]
    D = D.groupby(level=0).apply(lambda s: s.fillna(s.mean()))
    return D


def load_data(ret_sectors=False):
    sent = load_sent_data()
    min_date, max_date = (sent.index.levels[1].min(),
                          sent.index.levels[1].max())
    stock_ret = load_stock_data(ret_symbols=True,
                                ret_sectors=ret_sectors,
                                min_date=min_date,
                                max_date=max_date)
    trading_dates = mcal.get_calendar("NYSE")\
        .valid_days(min_date, max_date)

    stocks, symbols = stock_ret[0], stock_ret[1]
    sent = sent.loc[(symbols, slice(None)), :]
    D = pd.merge(left=sent, right=stocks,
                 left_index=True, right_index=True,
                 how="outer")\
                 .loc[(slice(None), trading_dates), :]
    D = repair_data(D)

    D = D.reset_index()\
        .pivot(index="Date", columns="Symbol", values=["sent", "Close"])\

    if ret_sectors:
        return (D, stock_ret[2])
    else:
        return D


def preprocess(D):
    D["Close"] = np.log(D["Close"])\
        .diff()\
        .replace({np.inf: np.nan,
                  -np.inf: np.nan})\

    D["sent"] = D["sent"].ewm(halflife=2.5)\
        .mean()\
        .diff()

    D = D.dropna()
    D = (D - D.mean()) / D.std()
    return D


def visualize_data():
    """
    Produce a bunch of visualizations of the stock data.
    """
    raise NotImplementedError
    D, symbols, sectors = load_data(ret_symbols=True,
                                    ret_sectors=True)
    N_symbols = len(symbols)
    sector_ordering = np.array([])
    for sector in sectors.index.levels[0]:
        sector_symbols = sectors.loc[sector].to_numpy()
        sector_ordering = np.hstack((sector_ordering,
                                     sector_symbols))

    # _plot_volatility_stats(D)
    # _check_stationarity

    def gc_lr_value(s1, s2):
        """
        Tests x2 --GC--> x1
        """
        X = pd.merge(s1, s2, left_index=True, right_index=True, how="outer")\
            .fillna(method="bfill")\
            .to_numpy()
        return grangercausalitytests(X, maxlag=1,
                                     verbose=False)[1][0]["lrtest"][0]

    GC = np.zeros((N_symbols, N_symbols))
    for i, c1 in enumerate(symbols):
        print("({} / {}): * --GC--> {}".format(i + 1, N_symbols, c1))
        for j, c2 in enumerate(symbols):
            if c1 == c2:
                GC[i, j] = -1
            else:
                GC[i, j] = gc_lr_value(D.loc[c1, "Close"],
                                       D.loc[c2, "Close"])

    # TODO: Ensure the right df is used here and above
    _GC = np.copy(GC)
    _GC[np.arange(N_symbols), np.arange(N_symbols)] = np.nan  # Ignore loops
    # GCx = xr.DataArray(sps.chi2.sf(GC, df=1),
    #                    coords=np.vstack((symbols, symbols)),
    #                    dims=["target_symbol", "driver_symbol"],
    #                    name="GC p-values")\
    #                    .to_dataset()
    GCx = xr.DataArray(GC,
                       coords=np.vstack((symbols, symbols)),
                       dims=["target_symbol", "driver_symbol"],
                       name="GC p-values")\
                       .to_dataset()

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(np.log(GCx["GC p-values"]))
    axes[1].imshow(np.log(GCx["GC p-values"].loc[sector_ordering, sector_ordering]))
    plt.show()
    return


def _check_stationarity(D):
    raise NotImplementedError
    # [adfuller(D[("Close", s)], regression="nc", autolag="BIC")[0] for s in D.columns.levels[1]]
    stationarity_stats = D.groupby(level=0)\
        .apply(lambda s: pd.Series(
            {"Close-adfuller": 
             adfuller(s["Close"].to_numpy(), regression="nc", autolag="BIC")[0],
             "Volume-adfuller":
             adfuller(s["Volume"].to_numpy(), regression="nc", autolag="BIC")[0]})
            )

    print(stationarity_stats.describe())

    # Significance table
    # {'1%': -3.432960050084045,
    #  '5%': -2.8626931078801285,
    #  '10%': -2.567383843706519},

    #        Close-adfuller  Volume-adfuller
    # count      443.000000       443.000000
    # mean       -63.486514       -23.999303
    # std         28.267345         2.053554
    # min       -199.969324       -33.322354
    # 25%        -74.175776       -25.284975
    # 50%        -57.379402       -24.376477
    # 75%        -46.980415       -22.881282
    # max         -8.436936       -18.611219
    return


def _plot_volatility_stats(D):
    D_stats = D.groupby(level=0)\
        .apply(lambda s: pd.Series({new_col: val for (new_col, val) in zip(
            chain.from_iterable(
                [col + "-nu", col + "-mu", col + "-std"] for col in s.columns),
            chain.from_iterable([sps.t.fit(s[col]) for col in s.columns]))}))

    fig, axes = plt.subplots(2, 1)
    sns.scatterplot(x="Close-nu", y="Close-std", data=D_stats, ax=axes[0])
    sns.scatterplot(x="Volume-nu", y="Volume-std", data=D_stats, ax=axes[1])
    axes[0].set_title("Closing Price")
    axes[0].set_xlabel("$\\nu$")
    axes[0].set_ylabel("$\\sigma^2$")

    axes[1].set_title("Daily Volume")
    axes[1].set_xlabel("$\\nu$")
    axes[1].set_ylabel("$\\sigma^2$")
    fig.suptitle("t-distribution $(\\sigma^2, \\nu)$"
                 "Volatility Statistics S\&P500 ({} - {})".format(
                     D.index.levels[1][0].year, D.index.levels[1][-1].year))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/volatility_stats.pdf", pad_inches=0)
    fig.savefig("figures/volatility_stats.png", pad_inches=0)
    plt.show()
    return
