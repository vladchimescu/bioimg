#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_dimred(X_df, dims='tsne',
                size=(9,8),
                title=None,
                font_scale=1.4,
                **kwargs):
    plt.figure(figsize = size)
    sn.set(font_scale=font_scale)
    sn.set_style('white')
    sn.despine()
    sn.scatterplot(data=X_df,
                   x = dims + '1',
                   y = dims + '2',
                   **kwargs)
    #plt.legend(loc='lower right', bbox_to_anchor=(1.2,0.05))
    plt.xlabel(dims.upper() + ' 1')
    plt.ylabel(dims.upper() + ' 2')
    if title is not None:
        plt.title(title)


def facet_dimred(X_df, nrows, ncols,
                 dims='tsne',
                 scale=5,
                 cmap=None,
                 cbaxes=None,
                 alpha=0.5):
    if cmap is None:
        cmap = sn.diverging_palette(240, 15, as_cmap=True)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize = (nrows*scale, ncols*scale))
    sn.despine()
    for r in range(nrows):
        for c in range(ncols):
            f = feat_subset[ncols*r+c]
            ax[r,c].set_title(f.replace('ch-', ''))
            maxval = np.max([np.abs(np.percentile(X_df[f].values, 0.1)),
                             np.abs(np.percentile(X_df[f].values, 0.9))])
            sc = ax[r,c].scatter(x=X_df[dims+'1'].values, 
                       y=X_df[dims+'2'].values,
                       c=X_df[f].values,
                           cmap=cmap, alpha=alpha,
                                 vmin=-maxval, vmax=maxval)
            if cbaxes is None:
                cbaxes = inset_axes(ax[r,c], width="3%", height="45%", loc=1) 
            plt.colorbar(sc, ax=ax[r,c], cax = cbaxes)
    fig.text(0.5, 0.04, dims.upper() + ' 1', ha='center')
    fig.text(0.04, 0.5, dims.upper() + ' 2', va='center', rotation='vertical')


def facet_density(X_df, feat_column,
                  col_wrap,
                  lw=3,
                  size=(10,10),
                  **kwargs):
    plt.figure(figsize=size)
    g = sn.FacetGrid(X_df,
                     col=feat_column,
                     col_wrap=col_wrap,
                     sharex=False, **kwargs)
    g.map(sn.kdeplot, "val", lw=lw).add_legend()
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        ax.set_title(feat_subset[i].replace('ch-', ''))
        ax.set_xlabel('')


def facet_boxplot(X_df, x, y,
                  feat_column,
                  ncols, nrows,
                  size=(10,10),
                  ticks_angle=90):
    plt.figure(figsize=size)
    g = sn.catplot(x=x, y=y, 
                   col=feat_column,
                   kind="box", data=X_df,
                   sharey=False,
                   col_wrap=ncols)
    axes = g.axes.flatten()
    plt.xticks(rotation=ticks_angle)
    for i, ax in enumerate(axes):
        ax.set_title(feat_subset[i].replace('ch-', ''))
        ax.set_xlabel('')
        if i % ncols == 0:
            ax.set_ylabel('Standardized value')
        if i > ((nrows - 1) * ncols - 1):
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
