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

    These are logged *after each epoch*, BUT each epoch may correspond to a
    different number of steps.  CAREFUL! Not all terms in the data frame match
    those from the debug log!  For example there is no `evaluation/Average
    Returns` in the data frame, even though it's in the debug log.

    I still have to understand what all these log terms *mean*. Here is my
    estimates:

    Exploration vs Evaluation: compute similar metrics, except former includes
    noise injected into the action based on some noise injection scheme,
    whereas the latter does not, and is just the DDPG policy. They are run for
    potentially a different number of steps. We have exploration steps per
    train loop, so if that value is 1000, and we run for some number of epochs
    (e.g., 1000) then each epoch is a train loop, I assume. So we get 1M
    exploration steps. But there can be a different number of *evaluation*
    steps per epoch. But I am seeing step counts that don't match, hopefully
    it's not too big of a deal.

    Rewards are near 0-1 as those are scaled in some way. We really want the
    *returns*, which correspond to results in the published literature.

    Questions:
    - Also, are these already with smoothing applied?
    - Contact, control, forward, survive?
    - Paths?
    """
    columns_plot = sorted([
        'exploration/Returns Mean',
        'exploration/num steps total',
        #'exploration/Num Paths',
        'evaluation/Returns Mean',
        'evaluation/num steps total',
        #'evaluation/Num Paths',
    ])
    progfile = join(args.path,'progress.csv')
    df = pd.read_csv(progfile, delimiter = ',')
    print("loaded csv, shape {}".format(df.shape))
    df_cols = sorted([column for column in df])
    for col in df_cols:
        print('  ',col)

    # One row per statistic.
    nrows, ncols = len(columns_plot), 1
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(14*ncols,4*nrows))
    title = args.title
    k = 10
    row = 0
    for column in df_cols:
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
    if title[-4:] == '.png':
        figname = '{}'.format(title)
    else:
        figname = '{}.png'.format(title)
    plt.savefig(figname)
    print("\nJust saved: {}".format(figname))


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('path', type=str)
    pp.add_argument('--title', type=str, default='tmp')
    args = pp.parse_args()
    plot(args)
