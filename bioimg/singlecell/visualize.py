#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ..base.plot import diverge_map


def plot_dimred(X_df, dims='tsne',
                size=(9,8),
                title=None,
                font_scale=1.4,
                **seaborn_args):
    '''Plot lower-dimensional embedding (t-SNE, UMAP, PCA)
       ---------------------------------------------------
       Plot high-dimensional data embedded in 2D. The function
       simply visualizes the pre-computed embedding, thus any
       dimension reduction can be visualized (t-SNE, UMAP, MDS, etc)

       Parameters
       ----------
       X_df : DataFrame
           DataFrame with dimension-reduced features in columns.
           The names should correspond to the embedding method
           as specified by `dims`, e.g. if `dims='tsne'` the columns
           should be 'tsne1' and 'tsne2'
       dims : string
           Embedding method (default='tsne')
       size : tuple (optional)
           Size of the figure. Default is (9,8)
       title : string (optional)
           Figure title
       font_scale : float (optional)
           Font size parameter. Default: 1.4
       **seaborn_args : optional named arguments
           Further arguments passed to seaborn.scatterplot, e.g.
           `hue` (variable name controlling point color) 
            or `style` (variable for point shape)
    '''
    fig, ax = plt.subplots(figsize = size)
    sn.set(font_scale=font_scale)
    sn.set_style('white')
    sn.despine()
    sn.scatterplot(data=X_df,
                   x = dims + '1',
                   y = dims + '2',
                   **seaborn_args)
    #plt.legend(loc='lower right', bbox_to_anchor=(1.2,0.05))
    plt.xlabel(dims.upper() + ' 1')
    plt.ylabel(dims.upper() + ' 2')
    if title is not None:
        plt.title(title)


def facet_dimred(X_df,
                 feat_subset,
                 nrows, ncols,
                 dims='tsne',
                 scale=5,
                 cmap=None,
                 alpha=0.5):
    '''Facetted embedding colored by feature values
       --------------------------------------------
       Plot high-dimensional data embedded in 2D and colored by
       image features specified by the user
       (e.g. area, eccentricity, shape features, etc can be used).
       The function simply visualizes the pre-computed embedding, thus any
       dimension reduction can be visualized (t-SNE, UMAP, MDS, etc)

       Parameters
       ----------
       X_df : DataFrame
           Input data with embedding vectors (2D embedding
           such as t-SNE or UMAP is expected) and additional
           features (such as mean_intensity, area, shape features, etc)
       feat_subset : array-like
           List of column names to plot
       nrows : int
           Number of rows in the plot matrix
       ncols : int
           Number of columns in the plot matrix
       dims : string
           Embedding method (default='tsne')
       scale : float (optional)
           Controls figure size which is computed as
           (nrows*scale, ncols*scale)
       cmap : colormap (optional)
           Default color map is divergent with higher values
           colored red and lower values blue, while zero is white
       alpha : float (optional)
           Transparency of points (Default: 0.5)
    '''
    if cmap is None:
        cmap = sn.diverging_palette(240, 15, as_cmap=True)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize = (ncols*scale, nrows*scale))
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
            cbaxes = inset_axes(ax[r,c], width="3%", height="45%", loc=1) 
            plt.colorbar(sc, ax=ax[r,c], cax = cbaxes)
    fig.text(0.5, 0.04, dims.upper() + ' 1', ha='center')
    fig.text(0.04, 0.5, dims.upper() + ' 2', va='center', rotation='vertical')


def facet_density(X_long,
                  ncols,
                  feat_column='feature',
                  lw=3,
                  size=(10,10),
                  **seaborn_args):
    '''Facetted density plots
       ----------------------
       Plot facetted distributions for selected image features
       
       Parameters
       ----------
       X_long : DataFrame
           Long DataFrame with named image features
           in the feature column `feat_column` and feature values
           in column 'val'
       feat_column : string
           Name of the column with feature keys. Default: 'feature'
       ncols: int
            Number of columns in the plot matrix
       lw : int (optional)
           Line width of density curves
       size : tuple (optional)
           Size of the figure. Default is (10,10)
       **seaborn_args : optional named arguments
            Further arguments passed to seaborn.scatterplot, e.g.
           `hue` (variable name controlling point color) 
            or `style` (variable for point shape)
    '''
    plt.figure(figsize=size)
    g = sn.FacetGrid(X_long,
                     col=feat_column,
                     col_wrap=ncols,
                     sharex=False, **seaborn_args)
    g.map(sn.kdeplot, "val", lw=lw).add_legend()
    feat_subset = X_long[feat_column].unique()
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        ax.set_title(feat_subset[i].replace('ch-', ''))
        ax.set_xlabel('')


def facet_boxplot(X_long, x, y,
                  ncols, nrows,
                  feat_column='feature',
                  size=(10,10),
                  ticks_angle=90,
                  **seaborn_args):
    '''Facetted boxplots
       -----------------
       Plot facetted boxplots of features stratified
       by a categorical variable (e.g. condition, drug, etc)

       Parameters
       ----------
       X_long : DataFrame
           Long DataFrame with named image features
           in the feature column `feat_column` and feature values
           in column 'val'
       x : string
           Column name holding the feature to be plotted on
           the x-axis
       y : string
           Column name holding the feature to be plotted on
           the y-axis
       ncols : int
           Number of columns in the plot matrix
       nrows : int
           Number of rows in the plot matrix
       feat_column : string
           Name of the column with feature keys. Default: 'feature'
       size : tuple (optional)
           Size of the figure. Default is (10,10)
       ticks_angle : float (optional)
           Angle of the x-axis tick labels. Default: 90 degrees
       **seaborn_args : optional named arguments
            Further arguments passed to seaborn.scatterplot, e.g.
           `hue` (variable name controlling point color) 
            or `style` (variable for point shape)**seaborn_args : optional named arguments
    '''
    plt.figure(figsize=size)
    g = sn.catplot(x=x, y=y, 
                   col=feat_column,
                   kind="box", data=X_long,
                   sharey=False,
                   col_wrap=ncols,
                   **seaborn_args)
    feat_subset = X_long[feat_column].unique()
    axes = g.axes.flatten()
    plt.xticks(rotation=ticks_angle)
    for i, ax in enumerate(axes):
        ax.set_title(feat_subset[i].replace('ch-', ''))
        ax.set_xlabel('')
        if i % ncols == 0:
            ax.set_ylabel('Standardized value')
        if i > ((nrows - 1) * ncols - 1):
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


def plot_heatmap(df,
                 xticklabels=False,
                 yticklabels=False,
                  cmap=None,
                 vmin=None, vmax=None,
                 size=(6.5,6),
                 cbar_pos=(1,.45,.03,.3),
                 row_dendro=False,
                 col_dendro=False):
    '''Plot clustered heatmap
       ----------------------
    
       Parameters
       ----------
       df : DataFrame
           Input DataFrame
       xticklabels : bool (optional)
           Show column labels (default=False)
       yticklabels : bool (optional)
           Show row labels (default=False)
       cmap : color map (optional)
           Defaults to a diverging color map
       vmin : float (optional)
           minimum value of the color scale
       vmax : float (optional)
           maximum value of the color scale
       size : tuple (optional)
           figure size
       cbar_pos : tuple (optional)
           Position of the color bar
       row_dendro : bool (optional)
           Show dendrogram for rows
       col_dendro : bool (optional)
           Show dendrogram for columns
    '''
    if cmap is None:
        cmap = diverge_map(low='teal', high='goldenrod')
    if vmin is None:
        vmin = df.to_numpy().min()
    if vmax is None:
        vmax = df.to_numpy().max()
    ax = sn.clustermap(df,
           xticklabels=xticklabels,
           yticklabels=yticklabels,
                  cmap=cmap,
                 cbar_pos=cbar_pos,
                  vmin=vmin,
                  vmax=vmax,
                  figsize=size)
    ax.ax_row_dendrogram.set_visible(row_dendro)
    ax.ax_col_dendrogram.set_visible(col_dendro)
