# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from io import BytesIO
import base64

def decomp_plot(df, time, signal, estimator, residue, name = None, max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(residue in df.columns)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df1 = df.tail(max_length);
    fig, axs = plt.subplots(ncols=2, figsize=(32, 16))
    df1.plot.line(time, [signal, estimator, residue], ax=axs[0] , grid = True)
    residues =  df1[residue].values
    lErrorStdDev = np.std(residues)
    #    axs[0].fill_between(time, df1[estimator].values - 2*lErrorStdDev,  df1[estimator].values + 2*lErrorStdDev, color='b', alpha=0.2)

    import scipy.stats as scistats
    scistats.probplot(residues, dist="norm", plot=axs[1])

    if(name is not None):
        fig.savefig(name + '_decomp_output.png')
        plt.close(fig)

def decomp_plot_as_png_base64(df, time, signal, estimator, residue, name = None, max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(residue in df.columns)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df1 = df.tail(max_length);
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    df1.plot.line(time, [signal, estimator, residue], ax=axs[0] , grid = True)
    residues =  df1[residue].values
    lErrorStdDev = np.std(residues)

    import scipy.stats as scistats
    scistats.probplot(residues, dist="norm", plot=axs[1])

    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue())
    plt.close(fig)
    return figdata_png.decode('utf8')
    

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

    # last value of the signal
    lLastSignalPos = df1[signal].dropna().tail(1).index;
    lEstimtorValue = df1[estimator][lLastSignalPos];
    df1.loc[lLastSignalPos , lower] = lEstimtorValue;
    df1.loc[lLastSignalPos , upper] = lEstimtorValue;

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=1, figsize=(16, 8))
    df1.plot.line(time, [signal, estimator, lower, upper], ax=axs, grid = True)

    x = df1[time];
    type1 = np.dtype(x)
    if(type1.kind == 'M'):
        x = x.apply(lambda t : t.date());
    axs.fill_between(x.values, df1[lower], df1[upper], color='blue', alpha=.5)

    if(name is not None):
        fig.savefig(name + '_prediction_intervals_output.png')
        plt.close(fig)
    

def prediction_interval_plot_as_png_base64(df, time, signal, estimator, lower, upper, name = None, max_length = 1000) :
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

    # last value of the signal
    lLastSignalPos = df1[signal].dropna().tail(1).index;
    lEstimtorValue = df1[estimator][lLastSignalPos];
    df1.loc[lLastSignalPos , lower] = lEstimtorValue;
    df1.loc[lLastSignalPos , upper] = lEstimtorValue;

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=1, figsize=(16, 8))
    df1.plot.line(time, [signal, estimator, lower, upper], ax=axs, grid = True)

    x = df1[time];
    type1 = np.dtype(x)
    if(type1.kind == 'M'):
        x = x.apply(lambda t : t.date());
    axs.fill_between(x.values, df1[lower], df1[upper], color='blue', alpha=.5)

    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    plt.close(fig)
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png.decode('utf8')


def qqplot_residues(df , residue):
    pass


def plot_hierarchy(structure , name):
    import pydot
    graph = pydot.Dot(graph_type='graph', rankdir='LR');
    lLevelsReversed = sorted(structure.keys(), reverse=True);
    for level in  lLevelsReversed:
        for col in structure[level].keys():
            node_col = pydot.Node(col, style="filled", fillcolor="red")
            graph.add_node(node_col);
            for col1 in structure[level][col]:
                node_col1 = pydot.Node(col1, style="filled", fillcolor="red")
                graph.add_node(node_col1);
                graph.add_edge(pydot.Edge(node_col, node_col1))
    if(name is not None):
        graph.write_png(name);
    else:
        from IPython.display import Image, display
        plot1 = Image(graph.create_png())
        display(plot1)
