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
    """Load the progress csv file, and plot.

    Plot:
      'exploration/Returns Mean',
      'exploration/num steps total',
      'evaluation/Returns Mean',
      'evaluation/num steps total',
    """
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(11*ncols,6*nrows))

    algorithms = sorted([x for x in os.listdir('data/') if args.env in x])
    assert len(algorithms) == 2
    colors = ['blue', 'red']

    for idx,alg in enumerate(algorithms):
        print('Currently on algorithm: ', alg)
        alg_dir = join('data', alg)
        progfiles = sorted([
                join(alg_dir, x, 'progress.csv') for x in os.listdir(alg_dir)
        ])
        expl_returns = []
        eval_returns = []
        expl_steps = []
        eval_steps = []

        for prog in progfiles:
            df = pd.read_csv(prog, delimiter = ',')

            expl_ret = df['exploration/Returns Mean'].tolist()
            expl_returns.append(expl_ret)
            eval_ret = df['evaluation/Returns Mean'].tolist()
            eval_returns.append(eval_ret)

            expl_sp = df['exploration/num steps total'].tolist()
            expl_steps.append(expl_sp)
            eval_sp = df['evaluation/num steps total'].tolist()
            eval_steps.append(eval_sp)

        expl_returns = np.array(expl_returns)
        eval_returns = np.array(eval_returns)
        xs = expl_returns.shape[1]
        expl_ret_mean = np.mean(expl_returns, axis=0)
        eval_ret_mean = np.mean(eval_returns, axis=0)
        expl_ret_std = np.mean(expl_returns, axis=0)
        eval_ret_std = np.mean(eval_returns, axis=0)

        w = 10
        label0 = '{} (w={}), lastavg {:.1f}'.format(
                    (alg).replace('rlkit-',''), w, np.mean(expl_ret_mean[-w:]))
        label1 = '{} (w={}), lastavg {:.1f}'.format(
                    (alg).replace('rlkit-',''), w, np.mean(eval_ret_mean[-w:]))
        ax[0,0].plot(np.arange(xs), smoothed(expl_ret_mean, w=w),
                     color=colors[idx], label=label0)
        ax[0,1].plot(np.arange(xs), smoothed(eval_ret_mean, w=w),
                     color=colors[idx], label=label1)

        # This can be noisy.
        if False:
            ax[0,0].fill_between(np.arange(xs),
                                 expl_ret_mean-expl_ret_std,
                                 expl_ret_mean+expl_ret_std,
                                 alpha=0.3,
                                 facecolor=colors[idx])
            ax[0,1].fill_between(np.arange(xs),
                                 eval_ret_mean-eval_ret_std,
                                 eval_ret_mean+eval_ret_std,
                                 alpha=0.3,
                                 facecolor=colors[idx])

    for i in range(2):
        ax[0,i].tick_params(axis='x', labelsize=ticksize)
        ax[0,i].tick_params(axis='y', labelsize=ticksize)
        leg = ax[0,i].legend(loc="best", ncol=1, prop={'size':legendsize})
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
    ax[0,0].set_title('{} (Exloration)'.format(args.env), fontsize=ysize)
    ax[0,1].set_title('{} (Evaluation)'.format(args.env), fontsize=ysize)

    plt.tight_layout()
    figname = 'fig-{}.png'.format(args.env)
    plt.savefig(figname)
    print("\nJust saved: {}".format(figname))


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('env', type=str)
    args = pp.parse_args()
    plot(args)
