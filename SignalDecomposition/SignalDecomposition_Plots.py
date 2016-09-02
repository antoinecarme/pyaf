import pandas as pd
import numpy as np

import pylab 
import scipy.stats as scistats

import matplotlib.pyplot as plt

def decomp_plot(df, time, signal, estimator, residue, max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(residue in df.columns)

    df1 = df.tail(max_length);
    fig, axs = plt.subplots(ncols=2, figsize=(32, 16))
    df1.plot.line(time, [signal, estimator, residue], ax=axs[0])
    residues =  df1[residue].values
    lErrorStdDev = np.std(residues)
    #    axs[0].fill_between(time, df1[estimator].values - 2*lErrorStdDev,  df1[estimator].values + 2*lErrorStdDev, color='b', alpha=0.2)
    scistats.probplot(residues, dist="norm", plot=axs[1])
    #    pylab.show(
    #plt.show()
    #plt.close()

def qqplot_residues(df , residue):
    pass
