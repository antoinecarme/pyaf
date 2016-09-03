import pandas as pd
import numpy as np

import pylab 
import scipy.stats as scistats

import matplotlib.pyplot as plt

def decomp_plot(df, time, signal, estimator, residue, name = None, max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(residue in df.columns)

    df1 = df.tail(max_length);
    fig, axs = plt.subplots(ncols=2, figsize=(32, 16))
    df1.plot.line(time, [signal, estimator, residue], ax=axs[0] , grid = True)
    residues =  df1[residue].values
    lErrorStdDev = np.std(residues)
    #    axs[0].fill_between(time, df1[estimator].values - 2*lErrorStdDev,  df1[estimator].values + 2*lErrorStdDev, color='b', alpha=0.2)
    scistats.probplot(residues, dist="norm", plot=axs[1])

    if(name is not None):
        fig.savefig(name + '_decomp_output.png')

def prediction_interval_plot(df, time, signal, estimator, lower, upper, name = None, max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(lower in df.columns)
    assert(upper in df.columns)


    df1 = df.tail(max_length).copy();
    lMin = np.mean(df1[signal]) -  np.std(df1[signal]) * 3;
    lMax = np.mean(df1[signal]) +  np.std(df1[signal]) * 3;
    df1[lower] = df1[lower].apply(lambda x : x if (np.isnan(x) or x >= lMin) else np.nan);
    df1[upper] = df1[upper].apply(lambda x : x if (np.isnan(x) or x <= lMax) else np.nan);
    fig, axs = plt.subplots(ncols=2, figsize=(32, 16))
    df1.plot.line(time, [signal, estimator, lower, upper], ax=axs[0], grid = True)
    axs[0].fill_between(df1[time], df1[lower], df1[upper], color='blue', alpha=.5)

    if(name is not None):
        fig.savefig(name + '_prediction_intervals_output.png')


def qqplot_residues(df , residue):
    pass
