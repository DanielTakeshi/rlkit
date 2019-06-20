"""Plot from the CSV from rlkit.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import argparse
import csv
import pandas as pd
import os
import sys
import pickle
import numpy as np
from os.path import join

# matplotlib
titlesize = 33
xsize = 30
ysize = 30
ticksize = 25
legendsize = 25
error_region_alpha = 0.25


def smoothed(x, w):
    """Smooth x by averaging over sliding windows of w, assuming sufficient length.
    """
    if len(x) <= w:
        return x
    smooth = []
    for i in range(1, w):
        smooth.append( np.mean(x[0:i]) )
    for i in range(w, len(x)+1):
        smooth.append( np.mean(x[i-w:i]) )
    assert len(x) == len(smooth), "lengths: {}, {}".format(len(x), len(smooth))
    return np.array(smooth)


def plot(args):
    """Load monitor curves and the progress csv file. And plot from those.

    TODO: still have to understand what all these *mean*, i.e. exploration vs
    evaluation? Does evaluation just mean some 'held out' trajectory not added
    to the replay buffer, but with no noise added to the actor?

    Also, are these already with smoothing applied?

    CAREFUL! Not all terms in the data frame match those from the debug log!
    For example there is no `evaluation/Average Returns` in the data frame,
    even though it's in the debug log.
    """
    columns_plot = sorted([
        'exploration/Returns Mean',
        'exploration/Rewards Mean',
        'evaluation/Returns Mean',
        'evaluation/Rewards Mean',
    ])
    progfile = join(args.path,'progress.csv')
    df = pd.read_csv(progfile, delimiter = ',')

    # (1000,118) for HalfCheetah example
    print("loaded csv, shape {}".format(df.shape))
    df_cols = sorted([column for column in df])
    for col in df_cols:
        print('  ',col)

    # One row per statistic.
    nrows, ncols = len(columns_plot), 1
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(13*ncols,4*nrows))
    title = args.title
    k = 10
    row = 0
    for column in df:
        if column not in columns_plot:
            continue
        data = df[column].tolist()
        label = 'avg all:  {:.2f}, last {}:  {:.2f}'.format(
                np.mean(data), k, np.mean(data[-k:]))
        ax[row,0].plot(data, label=label)
        ax[row,0].set_title(column, fontsize=ysize)
        ax[row,0].tick_params(axis='x', labelsize=ticksize)
        ax[row,0].tick_params(axis='y', labelsize=ticksize)
        leg = ax[row,0].legend(loc="best", ncol=1, prop={'size':legendsize})
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        row += 1

    plt.tight_layout()
    figname = '{}.png'.format(title)
    plt.savefig(figname)
    print("\nJust saved: {}".format(figname))


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('path', type=str)
    pp.add_argument('--title', type=str, default='tmp')
    args = pp.parse_args()
    plot(args)
