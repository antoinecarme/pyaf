# Copyright (C) 2016 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from io import BytesIO
import base64

from . import Utils as tsutil


SIGNAL_COLOR='green'
FORECAST_COLOR='blue'
RESIDUE_COLOR='red'
COMPONENT_COLOR='navy'
SHADED_COLOR='turquoise'
UPPER_COLOR='grey'
LOWER_COLOR='black'


def add_patched_legend(ax , names):
    # matplotlib does not like labels starting with '_'
    patched_names = []
    for name in names:
        # remove leading '_' => here, this is almost OK: no signal transformation
        patched_name = name.lstrip('_')
        patched_names = patched_names + [ patched_name ]
    # print("add_patched_legend" , names, patched_names)
    ax.legend(patched_names)

def fig_to_png_base64(fig):
    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png.decode('utf8')

def log_saving_plot_message(name, filename):
    logger = tsutil.get_pyaf_logger();
    logger.info("SAVING_PLOT " + str((name , filename)));
    
def decomp_plot_internal(df, time, signal, estimator, residue, name = None, format='png', max_length = 1000, horizon = 1, title = None) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(residue in df.columns)


    import matplotlib
    import matplotlib.pyplot as plt
    # print("MATPLOTLIB_BACKEND",  matplotlib.get_backend())
    # matplotlib.use('Agg')
    df1 = df.tail(max(max_length , 4 * horizon));
    if(name is not None):
        plt.switch_backend('Agg')
    fig, axs = plt.subplots(ncols=2, figsize=(32, 16))

    lColor = COMPONENT_COLOR;
    if(name is not None and name.endswith("Forecast")):
        lColor = FORECAST_COLOR;
    df1.plot.line(time, [signal, estimator, residue],
                  color=[SIGNAL_COLOR, lColor, RESIDUE_COLOR],
                  ax=axs[0] , grid = True, legend=False)
    add_patched_legend(axs[0] , [signal, estimator, residue])
    if(title is not None):
        axs[0].set_title(title + "\n")
    else:
        axs[0].set_title(estimator + "\n")
    residues =  df1[residue].values

    import scipy.stats as scistats
    resid = residues[~np.isnan(residues)]
    scistats.probplot(resid, dist="norm", plot=axs[1])

    return fig

def decomp_plot(df, time, signal, estimator, residue, name = None, format='png', max_length = 1000, horizon = 1, title = None) :
    fig = decomp_plot_internal(df, time, signal, estimator, residue, name, format, max_length, horizon, title)
    if(name is not None):
        import matplotlib
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        name1 = name.split('_')[-1]
        log_saving_plot_message(name1, name + '_decomp_output.' + format)
        fig.savefig(name + '_decomp_output.' + format)
        plt.close(fig)


def decomp_plot_as_png_base64(df, time, signal, estimator, residue, name = None, max_length = 1000, horizon = 1, title = None) :
    fig = decomp_plot_internal(df, time, signal, estimator, residue, name, format, max_length, horizon, title)
 
    import matplotlib
    import matplotlib.pyplot as plt
    png_b64 = fig_to_png_base64(fig)
    plt.close(fig)
    return png_b64
    
def prediction_interval_plot_internal(df, time, signal, estimator, lower, upper, name = None, format='png', max_length = 1000, horizon = 1, title = None) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(lower in df.columns)
    assert(upper in df.columns)


    df1 = df.tail(max(max_length, 4 * horizon)).copy();
    lMin = np.mean(df1[signal]) -  np.std(df1[signal]) * 10;
    lMax = np.mean(df1[signal]) +  np.std(df1[signal]) * 10;
    df1[lower] = df1[lower].apply(lambda x : x if (np.isnan(x) or x >= lMin) else np.nan);
    df1[upper] = df1[upper].apply(lambda x : x if (np.isnan(x) or x <= lMax) else np.nan);

    # last value of the signal
    lLastSignalPos = df1[signal].dropna().tail(1).index[0];
    lEstimtorValue = df1[estimator][lLastSignalPos];
    df1.loc[lLastSignalPos , lower] = lEstimtorValue;
    df1.loc[lLastSignalPos , upper] = lEstimtorValue;

    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if(name is not None):
        plt.switch_backend('Agg')
    fig, axs = plt.subplots(ncols=1, figsize=(16, 8))
    if(title is not None):
        axs.set_title(title + "\n")
    else:
        axs.set_title("Prediction Intervals\n")
        
    df1.plot.line(time, [signal, estimator, lower, upper],
                  color=[SIGNAL_COLOR, FORECAST_COLOR, LOWER_COLOR, UPPER_COLOR],
                  ax=axs, grid = True, legend=False)
    add_patched_legend(axs , [signal, estimator, lower, upper])

    axs.fill_between(df1[time].values, df1[lower], df1[upper], color=SHADED_COLOR, alpha=.2)

    return fig

def prediction_interval_plot(df, time, signal, estimator, lower, upper, name = None, format='png', max_length = 1000, horizon = 1, title = None) :
    fig = prediction_interval_plot_internal(df, time, signal, estimator, lower, upper, name, format, max_length, horizon, title)
    if(name is not None):
        import matplotlib
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        log_saving_plot_message('PredictionIntervals', name + '_prediction_intervals_output.' + format)
        fig.savefig(name + '_prediction_intervals_output.' + format)
        plt.close(fig)
    
def prediction_interval_plot_as_png_base64(df, time, signal, estimator, lower, upper, name = None, max_length = 1000, horizon = 1, title = None) :
    fig = prediction_interval_plot_internal(df, time, signal, estimator, lower, upper, name, format, max_length, horizon, title)

    import matplotlib
    import matplotlib.pyplot as plt
    png_b64 = fig_to_png_base64(fig)
    plt.close(fig)
    return png_b64


def quantiles_plot_internal(df, time, signal, estimator, iQuantiles, name = None, format='png', horizon = 1, title = None) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)

    lQuantileNames = [estimator + '_Quantile_' + str(q) for q in iQuantiles]
    df1 = df.tail(horizon)

    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if(name is not None):
        plt.switch_backend('Agg')
    lMin, lMax = df1[lQuantileNames].values.min(), df1[lQuantileNames].values.max()
    # Avoid a warning from matplotlib.
    lEps = 0.01
    if((lMax - lMin) < lEps):
        lMin, lMax = lMin - lEps, lMax + lEps
        
    #  Forecast Quantiles Plots can be improved #225 
    # Use a more meaningful color map (gradient, Blue = Low, Green = Normal, Red = High) for synchronized histograms.
    # Blue/Red for lower/higher quartile, decreasing alpha towards ther median.
    # Green for the 2nd and 3rd quartiles (inter-quartile range, not outliers)
    # pre-defined matplotlib colormap 'turbo' is the closest to this behavior. Don't reinvent the wheel.
    cm_turbo = matplotlib.colormaps.get('turbo')
    # quantize the colors (keep only 16)
    cm = matplotlib.colors.ListedColormap(cm_turbo.colors[0:256:16])
    #  Forecast Quantiles Plots can be improved #225
    # Better separate histograms (original issue solution). Assign a fixed height (1 cm) to each histogram.
    fig, axs = plt.subplots(horizon, 1, figsize=(12, horizon / 2.54), squeeze = True)
    # plt.subplots_adjust(hspace=1)
    # print(axs)
    if (horizon == 1):
        axs = [axs]
    for h in range(horizon):
        lIdx = df1.index[h]
        lTime = df1.loc[lIdx, time]
        q_values = df1.loc[lIdx, lQuantileNames].tolist()
        q_values = [max(x , -1e10) for x in q_values]
        q_values = [min(x , +1e10) for x in q_values]
        if((max(q_values) - min(q_values)) < lEps):
            # Avoid a warning from matplotlib for a constant signal.
            q_values = [min(q_values) - lEps] + [max(q_values) + lEps]
        # print(h, horizon, lIdx, lTime, q_values)
        _, bins1, patches = axs[h].hist(q_values, bins = q_values, weights=[1]*len(q_values), density = True)
        for i, p in enumerate(patches):
            j = (bins1[i] - lMin) / (lMax - lMin)
            plt.setp(p, 'facecolor', cm(j))
        if(h == 0):
            if(title is not None):
                axs[h].set_title(title)
            else:
                axs[h].set_title('Forecast Quantiles')
        axs[h].set_xlim((lMin,lMax))
        # axs[h].set_ylim((0, 1.0))

        # Remove some unnecessary borders and yticks.
        axs[h].spines['top'].set_visible(False)
        axs[h].spines['right'].set_visible(False)
        axs[h].spines['left'].set_visible(False)
        
        axs[h].set_ylabel('H=' + str(h + 1), rotation=0, horizontalalignment='left')
        axs[h].set_yticks([])
        axs[h].set_yticklabels([])
        if(h < (horizon - 1)):
            axs[h].set_xlabel('')
            axs[h].set_xticklabels([])

    return fig

def quantiles_plot(df, time, signal, estimator, iQuantiles, name = None, format='png', horizon = 1, title = None) :
    fig = quantiles_plot_internal(df, time, signal, estimator, iQuantiles, name, format, horizon, title)
    import matplotlib
    import matplotlib.pyplot as plt
    if(name is not None):
        plt.switch_backend('Agg')
        log_saving_plot_message('Quantiles', name + '_quantiles_output.' + format)
        fig.savefig(name + '_quantiles_output.' + format)
        plt.close(fig)
    

def quantiles_plot_as_png_base64(df, time, signal, estimator, iQuantiles, name = None, format='png', horizon = 1, title = None) :
    fig = quantiles_plot_internal(df, time, signal, estimator, iQuantiles, name, format, horizon, title)
    import matplotlib
    import matplotlib.pyplot as plt
    
    png_b64 = fig_to_png_base64(fig)
    plt.close(fig)
    return png_b64

def qqplot_residues(df , residue):
    pass

def build_record_label(labels_list):
    out = "<f0>" + str(labels_list[0]);
    i = 1;
    for l in labels_list[1:]:
        out = out + " | <f" + str(i) + "> " + str(l) ;
        i = i + 1;
    return out + "";


def plot_hierarchy_internal(structure , iAnnotations, name):
    import pydot
    graph = pydot.Dot(graph_type='graph', rankdir='LR', fontsize="12.0");
    graph.set_node_defaults(shape='record')
    lLevelsReversed = sorted(structure.keys(), reverse=True);
    for level in  lLevelsReversed:
        color = '#%02x%02x%02x' % (255, 255, 127 + int(128 * (1.0 - (level + 1.0) / len(lLevelsReversed))));
        for col in structure[level].keys():
            lLabel = col if iAnnotations is None else str(iAnnotations[col]);
            if iAnnotations is not None:
                lLabel = build_record_label(iAnnotations[col]);
            node_col = pydot.Node(col, label=lLabel, style="filled", fillcolor=color, fontsize="12.0")
            graph.add_node(node_col);
            for col1 in structure[level][col]:
                lLabel1 = col1
                if iAnnotations is not None:
                    lLabel1 = build_record_label(iAnnotations[col1]);
                color1 = '#%02x%02x%02x' % (255, 255, 128 + int(128 * (1.0 - (level + 2.0) / len(lLevelsReversed))));
                node_col1 = pydot.Node(col1, label=lLabel1, style="filled",
                                       fillcolor=color1, fontsize="12.0")
                graph.add_node(node_col1);
                lEdgeLabel = "";
                if iAnnotations is not None:
                    lEdgeLabel = iAnnotations[col + "_" + col1];
                lEdge = pydot.Edge(node_col, node_col1, color="red", label=lEdgeLabel, fontsize="12.0")
                graph.add_edge(lEdge)
    # print(graph.obj_dict)
    return graph
    
def plot_hierarchy(structure , iAnnotations, name):
    graph = plot_hierarchy_internal(structure , iAnnotations, name)
    if(name is not None):
        graph.write_png(name);
    else:
        from IPython.display import Image, display
        plot1 = Image(graph.create_png())
        display(plot1)


def plot_hierarchy_as_png_base64(structure , iAnnotations, name):
    graph = plot_hierarchy_internal(structure , iAnnotations, name)
    figdata_png = base64.b64encode(graph.create_png())
    return figdata_png.decode('utf8')

