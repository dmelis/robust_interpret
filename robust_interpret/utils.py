# -*- coding: utf-8 -*-
""" Code robustness evaluation in interpretability methods. Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os, sys
from tqdm import tqdm
import scipy as sp
import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
from skimage import feature, transform
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

def as_si(x, ndp):
    """ Convert humber to latex-style x10 scientific notation string"""
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def rgb2gray_converter(x):
    # x has shape (batch, width, height, channels)
    return (0.21 * x[:, :, :, :1]) + (0.72 * x[:, :, :, 1:2]) + (0.07 * x[:, :, :, -1:])

def topk_argmax(A, k):
    # Use argpartition to avoid sorting whole matrix
    argmaxs = np.argpartition(A,-k)[:,-k:]
    vals = np.array([A[k][idxs] for k,idxs in enumerate(argmaxs)])
    # We now have topk, but they're not sorted. Now sort these (fast since only k)
    argmaxs = np.array([argmaxs[i,np.argsort(-v)] for i,v in enumerate(vals)])
    vals = np.array([A[k][idxs] for k,idxs in enumerate(argmaxs)])
    return vals, argmaxs

def generate_dir_names(dataset, args, make=True):
    suffix = '{}_Eps{}_B{}'.format(
        args.optim,
        args.lip_eps,
        args.lip_calls,
    )
    model_path = os.path.join(args.model_path, dataset)
    log_path = os.path.join(args.log_path, dataset, suffix)
    results_path = os.path.join(args.results_path, dataset, suffix)

    if make:
        for p in [model_path, log_path, results_path]:
            if not os.path.exists(p):
                os.makedirs(p)

    model_path =  os.path.join(model_path, 'keras_cnn.h5')
    if args.debug:
        print(model_path, log_path, results_path)
        return model_path, '/tmp/', '/tmp/'
    return model_path, log_path, results_path


def plot_dataset(ax, ds_cnt, X_train, X_test, y_train, y_test, xlim, ylim):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(())
    ax.set_yticks(())


def plot_decision(ax, xx, yy, Z, X_train, X_test, y_train, y_test, score, title=None):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    if X_train is not None:
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
    if X_test is not None:
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if title:
        ax.set_title(title)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')


def plot_linear_explainer_multifeat(explainer, X, y, line_scale=1, plot_dims=None,
                                    title=None, ax=None, save_path=None):
    h = .02  # step size in the mesh
    cm = plt.cm.RdBu
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    if X.shape[1] == 2:
        d1, d2 = 0, 1
    elif plot_dims is not None:
        d1, d2 = plot_dims
    else:
        raise ValueError(
            'Warning: this method plots data in 2d. Have to choose dims to plot')

    x_min, x_max = X[:, d1].min() - .5, X[:, d1].max() + .5
    y_min, y_max = X[:, d2].min() - .5, X[:, d2].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    if X.shape[1] == 2:
        stacked_grid = np.c_[xx.ravel(), yy.ravel()]
    else:
        stacked_grid = np.tile(b[:, newaxis], (1, 100))

    #Z = explainer.model(stacked_grid)[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    coefs = explainer(X)[:, [d2, d1]]  # inverted because normal
    #coefs[:,[0, 1]] = coefs[:,[1, 0]]
    coefs[:, 1] *= -1
    ax.scatter(X[:, d1], X[:, d2], c=y, cmap=cm_bright, edgecolors='k', s=100)
    for i, center in enumerate(X):
        p1 = center - line_scale * coefs[i, :]
        p2 = center + line_scale * coefs[i, :]
        x_coord = [p1[0], p2[0]]
        y_coord = [p1[1], p2[1]]
        ax.plot(x_coord, y_coord, 'k-')
    if title:
        ax.set_title(title)

def plot_dependencies(dictionary_values,
                      pos_color="#ff4d4d",
                      negative_color="#3DE8F7",
                      reverse_values=False,
                      sort_rows =True,
                      scale_values = True,
                      title="",
                      fig_size=(4, 4),
                      ax = None,
                      x = None,
                      digits = 1, prediction_text = None,
                      show_table = False, ax_table = None):
    """ This function was adapted form the fairml python package
        x needed only if show_table = True
        digits: (int) significant digits in table
    """
    # add check to make sure that dependence features are not zeros
    if np.sum(np.array(dictionary_values.values())) == 0.0:
        print("Feature dependence for all attributes equal zero."
              " There is nothing to plot here. ")
        return None

    column_names = list(dictionary_values.keys())
    coefficient_values = list(dictionary_values.values())

    # get maximum
    maximum_value = np.absolute(np.array(coefficient_values)).max()
    if scale_values:
        coefficient_values = ((np.array(coefficient_values) / maximum_value) * 100)

    if sort_rows:
        index_sorted = np.argsort(np.array(coefficient_values))
    else:
        index_sorted = range(len(coefficient_values))[::-1]

    sorted_column_names = list(np.array(column_names)[index_sorted])
    sorted_column_values = list(np.array(coefficient_values)[index_sorted])
    pos = np.arange(len(sorted_column_values)) + 0.7

    # rearrange this at some other point.
    def assign_colors_to_bars(array_values,
                              pos_influence=pos_color,
                              negative_influence=negative_color,
                              reverse=reverse_values):

        # if you want the colors to be reversed for positive
        # and negative influences.
        if reverse:
            pos_influence, negative_influence = (negative_influence,
                                                 pos_influence)

        # could rewrite this as a lambda function
        # but I understand this better
        def map_x(x):
            if x > 0:
                return pos_influence
            else:
                return negative_influence
        bar_colors = list(map(map_x, array_values))
        return bar_colors

    bar_colors = assign_colors_to_bars(coefficient_values, reverse=True)
    bar_colors = list(np.array(bar_colors)[index_sorted])

    #pdb.set_trace()
    if ax is None and not show_table:
        #pdb.set_trace()
        fig, ax = plt.subplots(figsize=fig_size)
    elif ax is None and show_table:
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        ax_table, ax = axes

    ax.barh(pos, sorted_column_values, align='center', color=bar_colors)
    ax.set_yticks(pos)
    ax.set_yticklabels(sorted_column_names)
    if scale_values:
        ax.set_xlim(-105, 105)
    else:
        pass
        #ax.set_xlim(-1.05, 1.05)
    if title:
        ax.set_title("{}".format(title))

    if show_table and ax_table:
        cell_text = [[('%1.' + str(digits) + 'f') % v] for v in x]
        if prediction_text is None:
            ax_table.axis('off')
        else:
            print('here')
            ax_table.set_xticklabels([])
            ax_table.set_yticklabels([])
            ax_table.set_yticks([])
            ax_table.set_xticks([])
            for side in ['top', 'right', 'bottom', 'left']:
                ax_table.spines[side].set_visible(False)
            ax_table.set_xlabel(prediction_text)

        ax_table.table(cellText=cell_text,
                                  rowLabels=sorted_column_names[::-1],
                                  rowColours=bar_colors[::-1],
                                  colLabels=None,#['Value'],
                                  colWidths=[1],
                                  loc='left', cellLoc = 'right',
                                  bbox=[0.2, 0.025, 0.95, 0.95])
        ax_table.set_title('Input Value')
        return ax, ax_table

    return ax

def plot_importances(explainer, x, y, ax=None, feat=0):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    centers = x
    coefs = explainer(centers)
    length = 2
    print(coefs)
    importances = 100 * coefs[:, feat]
    print(importances)

    ax.set_title('Importances of Feature: {}'.format(feat))

    ax.scatter(centers[importances > 0, 0], centers[importances > 0, 1],
               marker='^', c=y[importances > 0], cmap=cm_bright,
               edgecolors='k', s=importances[importances > 0], )

    ax.scatter(centers[importances < 0, 0], centers[importances < 0, 1],
               marker='v', c=y[importances < 0], cmap=cm_bright,
               edgecolors='k', s=-importances[importances < 0], )


def plot_linear_explainer(explainer, X, y, line_scale=1, plot_dims=None, explain='predicted',
                          title=None, ax=None, save_path=None):
    h = .02  # step size in the mesh
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = explainer.model(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if (y is None) or (explain == 'predicted'):  # Will explain predicted class
        y_pred = np.argmax(explainer.model(X), axis=1)
        coefs = explainer(X, y_pred)
    else:  # Explain y
        coefs = explainer(X, y)

    if coefs.shape[1] > 2:
        # Probably means we have bias term
        coefs = coefs[:, :2]
    coefs[:, [0, 1]] = coefs[:, [1, 0]]
    coefs[:, 1] *= -1
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k', s=100)
    for i, center in enumerate(X):
        p1 = center - line_scale * coefs[i, :]
        p2 = center + line_scale * coefs[i, :]
        x_coord = [p1[0], p2[0]]
        y_coord = [p1[1], p2[1]]
        ax.plot(x_coord, y_coord, 'k-')
    if title:
        ax.set_title(title)




def barchart_insert(coefs, ax=None, labels=None):
    if ax is None:
        fig, ax = plt.subplots()
    N = 2
    width = 0.9
    ind = np.arange(N)  # the x locations for the groups
    cmap = matplotlib.cm.get_cmap('tab10')
    # color = ['r','b'])
    ax.bar(ind + width, coefs, width, color=[cmap(2), cmap(4)])
    ax.set_xticks(ind + width)
    if labels:
        ax.set_xticklabels(labels)
    else:
        ax.set_xticklabels([])
        ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_linewidth(2)


def plot_relevance_explainer(explainer, X, y, line_scale=1, plot_dims=None, explain='predicted',
                             title=None, ax=None, div_x=False, point_size=50, save_path=None):
    """
        div_x: devide attribution coefficient by value of x, useful for models like SHAP theta_reg_type
               essentially return x_i*b_i

    """
    h = .02  # step size in the mesh
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = explainer.model(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.7)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax.set_xticks(())
    # ax.set_yticks(())
    if (y is None) or (explain == 'predicted'):  # Will explain predicted class
        y_pred = np.argmax(explainer.model(X), axis=1)
        coefs = explainer(X, y_pred)
    else:  # Explain y
        coefs = explainer(X, y)
    axis_to_data = ax.transAxes + ax.transData.inverted()
    data_to_axis = axis_to_data.inverted()

    if coefs.shape[1] > 2:
        # Probably means we have bias term
        coefs = coefs[:, :2]

    if div_x:
        coefs /= (X + 1e-6)
    #coefs[:,[0, 1]] = coefs[:,[1, 0]]
    #coefs[:,1] *= -1
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
               edgecolors='k', s=point_size)
    for i, center in enumerate(X):
        iax = plt.axes([0, 0, 1, 1], label=title + str(i))
        posx, posy = data_to_axis.transform(center - [.1, 0])
        #print(posx, posy)
        # posx, posy, width, height
        ip = InsetPosition(ax, [posx, posy, 0.05, 0.05])
        iax.set_axes_locator(ip)
        iax.patch.set_alpha(0)
        barchart_insert(coefs[i, :], ax=iax)

    if title:
        ax.set_title(title, fontsize=20)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)


def deepexplain_plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(
            xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges
    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max


    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='nearest',
                cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='nearest',
                    cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis


def plot_prob_drop(attribs, prob_drop, save_path = None):

    ind = np.arange(len(attribs))
    column_names = [str(j) for j in range(1,22)]

    width = 0.65

    fig, ax1 = plt.subplots(figsize=(8,4))

    color1 = '#377eb8'
    ax1.bar(ind+width+0.35, attribs, 0.45, color=color1)
    ax1.set_ylabel(r'Feature Relevance $\theta(x)_i$',color=color1, fontsize = 14)
    #ax1.set_ylim(-1,1)
    ax1.set_xlabel('Feature')
    ax1.tick_params(axis='y', colors=color1)


    color2 = '#ff7f00'
    ax2 = ax1.twinx()
    ax2.ticklabel_format(style='sci',scilimits=(-2,2),axis = 'y')
    ax2.plot(ind+width+0.35, prob_drop, 'bo', linestyle='dashed', color=color2)
    ax2.set_ylabel('Probability Drop', color = color2, fontsize = 14)
    ax2.tick_params(axis='y', colors=color2)


    ax1.set_xticks(ind+width+(width/2))
    ax1.set_xticklabels(column_names)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches = 'tight', format='pdf', dpi=300)

    plt.show()



def plot_attribution_stability(model, explainers, input, pert_type='gauss', noise_level=0.5,
                               samples=5, greys = False, show_proba=False, save_path=None, layout='vertical'):
    """ Test stability of relevance scores theta for perturbations of an input.

        Visualizes the perturbations of dependencies with respect to predicted class.

        Args:
            model (model): scikit-learn compatible model
            inputs (list of tensors): Inputs over which stability will be tested. First one is "base" input.
            explainers: either a single explainer or a dict of explainers

        Returns:
            stability: scalar.

        Displays plot also.

    """
    def gauss_perturbation(x, scale=1, vmin=None, vmax=None):
        noise = scale * np.random.randn(*x.shape)
        pert = x + noise
        if vmin:
            pert = pert.clip(min=vmin, max=vmax)
        return pert

    if type(explainers) is not dict:
        explainers = {'Exp': explainers}

    # Generate perturbations
    vmin, vmax = input.min(), input.max()
    inputs = [input]
    for i in range(samples):
        inputs.append(gauss_perturbation(
            input, scale=noise_level, vmin=vmin, vmax=vmax))

    inputs = np.stack(inputs, 0)
    outputs = model.predict(inputs)
    preds = np.argmax(outputs, 1)

    atts = {}
    for k, expl in explainers.items():
        expl_atts = expl(inputs, None, show_plot=False).reshape(*inputs.shape)
        if inputs.shape[-1] == 3:
            # for color images, aggregate attribs ove channels
            expl_atts = expl_atts.mean(-1)
        atts[k] = expl_atts

    ncol = samples + 1
    nrow = len(explainers.keys()) + 1
    if layout == 'horizontal':
        ncol = len(explainers.keys()) + 1
        nrow = samples + 1

    fig, ax = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))

    # Plot Inputs
    for i, x in enumerate(inputs):
        pos = (0, i) if layout == 'vertical' else (i, 0)
        #pred = model.predict(x)
        # , cmap = 'Greys', interpolation = 'nearest')
        if greys:
            ax[pos].imshow(x.squeeze(), vmin=vmin, vmax=vmax,
                       interpolation='nearest', cmap='Greys')
        else:
            ax[pos].imshow(x.squeeze(), vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        ax[pos].set_xticks([])
        ax[pos].set_yticks([])
        ax[pos].axis('off')
        if i == 0:
            ax[pos].set_title('Original'.format(i), fontsize=20)
        else:
            if show_proba:
                title = 'P({})={:0.4e}'.format(preds[i], outputs[i, preds[i]])
            else:
                title = 'Perturbation {}'.format(i)
            ax[pos].set_title(title, fontsize=20)

    # Plot Attributions
    for i, (method, att) in enumerate(atts.items()):
        print(method)
        for j in range(samples + 1):
            pos = (i + 1, j) if layout == 'vertical' else (j, i + 1)
            axij = deepexplain_plot(att[j].reshape(
                x.shape[0], x.shape[1]), xi=input, axis=ax[pos], alpha=0.5)  # .set_title('Attributions')
            if j == 0:
                axij.set_yticklabels([])
                axij.set_xticklabels([])
                axij.set_yticks([])
                axij.set_xticks([])
                if layout == 'vertical':
                    axij.set_ylabel(method)  # , fontsize = 20)
                    for side in ['top', 'right', 'bottom', 'left']:
                        axij.spines[side].set_visible(False)
                else:
                    axij.axis('off')
                    axij.set_title(method, fontsize=20)
            else:
                d = np.linalg.norm(att[j] - att[0])
                axij.set_title(r"$\hat{{L}}={:4.2f}$".format(d), fontsize=20)
    # fig.subplots_adjust(wspace=0.2)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.2)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show(block=False)


def lipschitz_boxplot(order=None, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    if order:
        sns.boxplot(data=df, ax=ax, palette="Set2",
                    order=order)  # , fontsize=14)
    else:
        sns.boxplot(data=df, ax=ax, palette="Set2")
    if continuous:
        ax.set_ylabel('Relative Cont. Lipschitz Estimate')
    else:
        ax.set_ylabel('Relative Disc. Lipschitz Estimate')
    #ax.set_ylabel(r'$\sup_{x\in B_{\epsilon}(x_0)} \frac{ \|\theta(x) - \theta(x_0) \|}{\|h(x) - h(x_0) \|}$', fontsize=20)
    ax.set_xlabel(r'Intepretation Method')
    labels = [k for k in all_metrics.keys()]
    ax.set_xticklabels(labels, rotation=45)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()


def lipschitz_argmax_plot(x, y, att_x, att_y, lip=None, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(3 * 2, 3 * 2))
    xi = x.squeeze().copy()
    xi = (xi - np.min(xi))
    xi /= np.max(xi)

    vmin, vmax = xi.min(), xi.max()
    ax[0, 0].imshow(xi, vmin=vmin, vmax=vmax, interpolation='nearest')

    ax[0, 0].set_title('Original')
    ax[0, 0].axis('off')
    deepexplain_plot(att_x.reshape(xi.shape), xi=x, alpha=0.3,
                     axis=ax[1, 0])  # .set_title('Attributions')
    yi = y.squeeze().copy()
    yi = (yi - np.min(yi))
    yi /= np.max(yi)
    ax[0, 1].imshow(yi, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax[0, 1].set_title('Perturbed')
    ax[0, 1].axis('off')
    deepexplain_plot(att_y.reshape(xi.shape), xi=x, alpha=0.3,
                     axis=ax[1, 1])  # .set_title('Attributions')
    if lip:
        ax[1, 1].set_title(r'$\delta={:4.2e}$'.format(lip))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show(block=False)

def lipschitz_feature_argmax_plot(x, y, att_x, att_y, pred_x = None, pred_y =None,
                                  feat_names = None, lip=None,
                                  figsize = (4,6), widths = (1,4),
                                  digits = 1, ax=None, save_path=None):
    """
        If prediction values pred_x, pred_y are passed, argmax is included in title
    """
    if ax is None:
        print(list(widths))
        fig, ax = plt.subplots(2, 2, figsize=figsize,  gridspec_kw = {'width_ratios':list(widths)})

    xmax = max(att_x.max(), att_y.max()) + 0.1
    xmin = min(att_x.min(), att_y.min()) - 0.1

    exp_dict_x = dict(zip(feat_names, att_x))
    xtext, ytext = None, None
    if pred_x is not None:
        lab_x = int(np.argmax(pred_x))
        xtext = r'$P(y={:})={:.2f}$'.format(lab_x, pred_x[lab_x])
    if pred_y is not None:
        lab_y = int(np.argmax(pred_y))
        ytext = r'$P(y={:})={:.2f}$'.format(lab_y, pred_y[lab_y])

    _ = plot_dependencies(exp_dict_x, x = x, sort_rows = False, scale_values = False, ax = ax[0,1],
                            show_table = True,digits=digits, ax_table = ax[0,0],
                            prediction_text = xtext, title = 'Explanation')
    ax[0,1].set_xlim(xmin, xmax)

    exp_dict_y = dict(zip(feat_names, att_y))
    _ = plot_dependencies(exp_dict_y, x = y, sort_rows = False, scale_values = False, ax = ax[1,1],
                            show_table = True, digits = digits, ax_table = ax[1,0],
                            prediction_text = ytext, title = 'Explanation')
    ax[1,1].set_xlim(xmin, xmax)

    #plt.subplots_adjust(bottom=0.0, left = 0.2)
    plt.subplots_adjust(wspace=None, hspace=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show(block=False)



def attrib_consistency_plot(df, xvar, yvar, xerrvar=None, yerrvar=None,
                    figsize=(6,5), title=None, hue='dataset',
                    show_correlation=True, corrtype='pearson', sci_pval=True,
                    annotate=True, annotation_arrows=True, annotation_fontsize=12,
                    force_text=0.5,
                    legend_fontsize=12,
                    title_fontsize=12,
                    marker_size=12,
                    arrowcolor='gray',
                    barcolor='gray',
                    xlabel = 'Attribution Score',
                    ylabel = r'Prediction Probability Change',
                    hue_legend=False,
                    color='#1f77b4',
                    lw=1,
                    xlim=None,
                    ylim=None,
                    ax=None,
                    show=True,
                    save_path=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    #### Compute Correlation
    if show_correlation:
        if corrtype == 'spearman':
            corr, p = spearmanr(df[xvar], df[yvar])
            corrsymbol = '\\rho'
        elif corrtype == 'pearson':
            corr, p = pearsonr(df[xvar], df[yvar])
            corrsymbol = 'r'
        else:
            raise ValueError('Unrecognized correlation type')
        if p < 0.01 and sci_pval:
            legend_label = r"${}: {:2.2f}$".format(corrsymbol,corr) + "\n" + r"p-value: ${:s}$".format(as_si(p,1))
        else:
            legend_label = r"${}: {:2.2f}$".format(corrsymbol,corr) + "\n" + r"p-value: ${:2.2f}$".format(p)
    else:
        legend_label = None


    ### Actual Plots - First does scatter only, second does line
    if hue == 'dataset': # Regplot doesn't take hue arg, so plot point separaetly
        sns.scatterplot(xvar,yvar, hue=hue, cmap='Set',data=df, ax = ax)#size=marker_size,
        if hue_legend:
            scatter_legend = ax.legend(loc='best')

        sns.regplot(x=xvar, y=yvar, data=df, ax = ax, color=color, label=legend_label,
                scatter_kws={'alpha': 0.0},
                line_kws={'lw': lw}
                )

        H, L = ax.get_legend_handles_labels()
        _line_handles = [ax.get_lines()[0]], [L[-1]]
        #ax.legend([ax.get_lines()[0]], ax.get_legend_handles_labels()[-1],handlelength=1.0,loc='best')#, handletextpad=0.0)


        #line_legend = ax.legend(loc='best')
        #h, l = ax.get_legend_handles_labels()  # contains both line and point labels
        #pdb.set_trace()
        ax.legend()
    else:
        sns.regplot(x=xvar, y=yvar, data=df, ax = ax, color=color, label=legend_label,
                scatter_kws={'s':marker_size},
                line_kws={'lw': lw}
                )
    #
    #
    # ### Add Error Bars
    # if xerrvar or yerrvar:
    #     xerr = df[xerrvar] if xerrvar else None
    #     yerr = df[yerrvar] if yerrvar else None
    #     #ax.errorbar(df[xvar], df[yvar], xerr=xerr, yerr=yerr, color=barcolor, fmt='none')
    #     ax.errorbar(df[xvar], df[yvar], xerr=xerr, yerr=yerr, fmt='none', ecolor='#d6d4d4',  alpha=0.75,elinewidth=0.75)

    # ### Annotate Points
    # if annotate:
    #     texts = []
    #     for i,a in df.iterrows():
    #         lab = r'{}$\rightarrow${}'.format(a.src,a.tgt) if a.tgt is not None else r'{}'.format(a.src)
    #         texts.append(ax.text(a[xvar], a[yvar], lab,fontsize=annotation_fontsize))
    #     #texts= [ax.text(a[xvar],a[yvar],, ) for i,a in df.iterrows()]
    #     if annotation_arrows:
    #         adjust_text(texts, force_text=force_text, arrowprops=dict(arrowstyle="-", color=arrowcolor, alpha=0.5, lw=0.5))
    #     else:
    #         adjust_text(texts, force_text=force_text)

    ### Fix Legend for Correlation (otherwise don't show)
    if show_correlation:
        plt.rc('legend',fontsize=legend_fontsize)#,borderpad=0.2,handletextpad=0, handlelength=0) # using a size in points
        #pdb.set_trace()
        #H, L = ax.get_legend_handles_labels()
        #ax.legend([ax.get_lines()[0]], ax.get_legend_handles_labels()[-1],handlelength=1.0,loc='best')#, handletextpad=0.0)
        #line_legend = ax.legend([ax.get_lines()[0]], [L[-1]],handlelength=1.0,loc='best')#, handletextpad=0.0)
        line_legend = ax.legend(*_line_handles,handlelength=1.0,loc='best')#, handletextpad=0.0)
        if hue_legend: ax.add_artist(scatter_legend)


    ### Add title and labels
    ax.set_xlabel(xlabel, fontsize=title_fontsize)
    ax.set_ylabel(ylabel, fontsize=title_fontsize)
    ax.set_title(r'Attribution Score Consistency' + (': {}'.format(title) if title else ''), fontsize=title_fontsize)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)


    #fig.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches = "tight")

    if show: plt.show()

    return ax
