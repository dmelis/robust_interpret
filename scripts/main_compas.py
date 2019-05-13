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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
# Necessary for parallelism in joblib + keras
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
# 0: all logs shown, 1: filter out INFO logs 2: filter out WARNING logs, and 3 to additionally filter out ERROR log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tempfile
import sys
#sys.path.insert(0, os.path.abspath('..'))

import robust_interpret
import scipy


import warnings
from functools import partial
import numpy as np
import pandas as pd
import matplotlib
    # Hack to be able to save plots in janis
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import pickle
from tqdm import tqdm
import argparse


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata

# Local imports
from os.path import dirname, realpath
sys.path.append(os.path.join(dirname(realpath(__file__)), 'codebase/'))

#import utils
#import explainers

from robust_interpret.explainers import lime_wrapper, shap_wrapper
from lime.wrappers.scikit_image import SegmentationAlgorithm

#from utils import deepexplain_plot, lipschitz_boxplot, lipschitz_argmax_plot
from robust_interpret.utils import generate_dir_names, plot_attribution_stability  # plot_theta_stability,
from robust_interpret.utils import lipschitz_feature_argmax_plot


def find_conflicting(df, labels, consensus_delta = 0.2):
    """
        Find examples with same exact feat vector but different label.
        Finds pairs of examples in dataframe that differ only
        in a few feature values.

        Args:
            - differ_in: list of col names over which rows can differ
    """
    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in tqdm(range(len(df))):
        if full_dups[i] and (not i in all_seen):
            i_dups = finder(df, df.iloc[i])
            groups.append(i_dups.index)
            all_seen.update(i_dups.index)

    pruned_df  = []
    pruned_lab = []
    for group in groups:
        scores = np.array([labels[i] for i in group])
        consensus = round(scores.mean())
        for i in group:
            if (abs(scores.mean() - 0.5)< consensus_delta) or labels[i] == consensus:
                # First condition: consensus is close to 50/50, can't consider this "outliers", so keep them all
                #print(scores.mean(), len(group))
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)


def load_compas_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size=64):
    df= pd.read_csv("/Users/david/pkg/fairml/doc/example_notebooks/propublica_data_for_fairml.csv")
    # Binarize num of priors var? Or normalize it 0,1?
    df['Number_of_Priors'] = np.sqrt(df['Number_of_Priors'])/(np.sqrt(38))
    compas_rating = df.score_factor.values # This is the target??
    df = df.drop("score_factor", 1)

    pruned_df, pruned_rating = find_conflicting(df, compas_rating)
    #pruned_df, pruned_rating = df, compas_rating # REOMVE AFRTWER DEBUG

    x_train, x_test, y_train, y_test   = train_test_split(pruned_df, pruned_rating, test_size=0.1, random_state=85)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=85)

    feat_names = list(x_train.columns)
    x_train = x_train.values # pandas -> np
    x_test  = x_test.values

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), feat_names

def parse_args():
    parser = argparse.ArgumentParser(
        description='Interpteratbility robustness evaluation on MNIST')

    # #setup
    parser.add_argument('--train', action='store_true',
                        default=False, help='Whether or not to train model')
    parser.add_argument('--metric', type=str, default='chebyshev',
                        help='metric for computing || x - y|| in discrete Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.1,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    # parser.add_argument('--lip_points', type=int, default=100,
    #                     help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    # #paths
    parser.add_argument('--model_path', type=str,
                        default='checkpoints', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='out',
                        help='where to dump model config and epoch Results')
    parser.add_argument('--log_path', type=str, default='log',
                        help='where to dump training logs  epoch Results (and config??)')
    parser.add_argument('--summary_path', type=str, default='results/summary.csv',
                        help='where to dump model config and epoch Results')

    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode (verbose, no saving)')

    args = parser.parse_args()

    # # update args and print
    # args.filters = [int(k) for k in args.filters.split(',')]
    # if args.objective == 'mse':
    #     args.num_class = 1

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


def main():
    args = parse_args()
    print(" ******* Remember to revert to *4 calls in (2) !!!!!!!!!")
    # DATA LOADING
    model_path, log_path, results_path = generate_dir_names(
        'compas', args, make=not args.debug)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test), feat_names = load_compas_data()

    # # Train simple model
    classif = LogisticRegression(penalty='l2', C=0.01)
    classif.fit(x_train, y_train)
    print('Random Forest Classif Accuracy {:4.2f}'.format(100*classif.score(x_test, y_test)))

    LIME = lime_wrapper(classif.predict_proba,
                    lime_type = 'tabular',
                    mode      = 'classification',
                    multiclass = True,
                    feature_names = list(feat_names),
                    class_names   = ['Negative', 'Positive'],
                    train_data       = x_train,
                    num_samples  = 100,
                    verbose = False)

    SHAP = shap_wrapper(classif.predict_proba,
                        shap_type = 'kernel',
                        link      = 'identity',
                        mode      = 'classification',
                        multiclass = True,
                        feature_names = list(feat_names),
                        class_names   = ['Negative', 'Positive'],
                        train_data       = x_train,
                        nsamples  = 100,
                        verbose = False)

    explainer_dict = {
        'LIME': LIME,
        'SHAP': SHAP
    }


    DiscLip = {k: {} for k in explainer_dict.keys()}

    for k, expl in explainer_dict.items():
        # print(k)
        # if k == 'LIME': continue
        # if k == 'SHAP': x_test = x_test[:50]
        vals, argmaxes = expl.estimate_discrete_dataset_lipschitz(x_test,
                        top_k=3, metric = args.metric, same_class = True)

        max_lip = vals.max()
        imax, _ = np.unravel_index(np.argmax(vals), vals.shape)
        jmax = argmaxes[imax][0][0]

        print('Max Lip value: {}, attained for pair ({},{})'.format(max_lip, imax, jmax))

        x      = x_test[imax]
        argmax = x_test[jmax]

        att_x      = expl(x, None, show_plot=False)
        att_argmax = expl(argmax, None, show_plot=False)

        DiscLip[k] = {'vals': vals, 'argmaxes': argmaxes, 'x': x, 'argmax': argmax}
        fpath = os.path.join(results_path, 'discrete_lip_{}').format(k)
        ppath = os.path.join(results_path, 'relevance_argmax_{}').format(k)
        pickle.dump(DiscLip[k], open(fpath+'.pkl',"wb"))
        lipschitz_feature_argmax_plot(x, argmax, att_x, att_argmax,
                                      feat_names = expl.feature_names,
                                      digits=2, figsize=(5,6), widths=(2,3),
                                      save_path=ppath + '.pdf')

    pickle.dump(DiscLip, open(results_path + '/discrete_lip_combined.pkl', "wb"))

    print('Done!')

if __name__ == '__main__':
    main()
