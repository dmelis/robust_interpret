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

import os
import pdb
import glob
# Necessary for parallelism in joblib + keras
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
# 0: all logs shown, 1: filter out INFO logs 2: filter out WARNING logs, and 3 to additionally filter out ERROR log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tempfile
from tqdm.autonotebook import tqdm

import sys
sys.path.insert(0, os.path.abspath('.'))
import warnings
from functools import partial
import scipy as sp
import numpy as np
import matplotlib
# Hack to be able to save plots in remote machine:
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import argparse
import collections

import sklearn.impute
import sklearn.datasets
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import fetch_openml #fetch_mldata

from lime.wrappers.scikit_image import SegmentationAlgorithm

# Local imports
#from os.path import dirname, realpath
#sys.path.append(os.path.join(dirname(realpath(__file__)), 'robust_interpret/'))
import robust_interpret
from robust_interpret.explainers import lime_wrapper, shap_wrapper, woe_wrapper
from robust_interpret.utils import generate_dir_names  # plot_theta_stability,
from robust_interpret.utils import plot_attribution_stability
from robust_interpret.utils import lipschitz_feature_argmax_plot
from robust_interpret.utils import plot_prob_drop

def check_completed_datasets(results_path):
    existing = list(glob.iglob(results_path + '/*.pkl'))
    done = set([os.path.splitext(os.path.basename(fpath))[0].split('_')[0] for fpath in existing])
    return done

def load_uci_data(dataname, verbose=False):
    #dataset = fetch_mldata(dataname, target_name='label', data_name='data',transpose_data=True)
    dataset = sklearn.datasets.fetch_openml(dataname)#, target_column='label')#, transpose_data=True)
    if verbose: print(dataset['DESCR'])
    feat_names = dataset['feature_names']
    y = dataset['target']
    if sp.sparse.issparse(y):
        y = np.array(y.todense().transpose())
        y = np.argmax(y, axis = 1).flatten()

    X = dataset.data
    if sp.sparse.issparse(X):
        X = np.array(dataset.data.todense())

    if np.isnan(X).any():
        # Impute Nans
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X = imp.fit_transform(X)

    # Get rid of any label that only shows up once
    counts = collections.Counter(y)
    filter = sorted([c for c,freq in counts.items() if freq >= 2])
    X = X[np.in1d(y, filter),:]
    y = y[np.in1d(y, filter)]

    feat_names = ['X'+ str(i) for i in range(dataset.data.shape[1])]
    dataset.feat_names = feat_names
    classes, y_int = np.unique(y, return_inverse=True)
    classes.sort()

    binary = len(set(classes)) == 2
    if binary and min(classes) == -1:
        y[y==-1] = 0
    elif binary and (set(classes) != set([0, 1])):
        y[y==classes[0]] = 0
        y[y==classes[1]] = 1
    else:
        if min(classes) == 1:
            y -= 1
            classes -= 1

    if type(classes[0]) == str: # some datasets have string target values
        y = y_int

    y = y.astype('int32')

    # Had to remove this becayse scikit learn classifier estimatesr classes based
    # on actual labels in train data
    # if int(y.max() + 1) != len(classes):
    #     #Some classes not represented in dataset. Cheap fix: pretend they do exist, but haven't seen dthe,
    #     classes = list(range(int(y.max() + 1)))


    nclasses = 1 if binary else len(classes)

    if len(set(classes)) == 2:
        if np.issubdtype(classes.dtype, np.number):
            class_names = ['Negative', 'Positive']
        else:
            class_names = classes.tolist()
        binary = True
    else:
        if np.issubdtype(classes.dtype, np.number):
            class_names = ['C' + str(i) for i in range(len(classes))]
        else:
            class_names = classes.tolist()
        binary = False

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
    # Stratified seems to be broken
    # sss = StratifiedShuffleSplit(n_splits = 2,test_size =0.1)
    # train_idx, test_idx = sss.split(X, y)
    # x_train, y_train = zip(*[(X[i], y[i]) for i in train_idx])
    # x_train, y_train = zip(*[(X[i], y[i]) for i in train_idx])
    return (x_train, y_train), (x_test, y_test), feat_names, class_names


def train_classifier(x_train, y_train, x_test, y_test, classes):
    classif = RandomForestClassifier(n_estimators=1000)#, class_weight=class_weight)
    classif.fit(x_train, y_train)
    print('Random Forest Classif Accuracy {:4.2f}%'.format(100*classif.score(x_test, y_test)))
    return classif

def parse_args():
    parser = argparse.ArgumentParser(
        description='Interpteratbility robustness evaluation on UCI Datasets')

    ### Setup
    parser.add_argument('--train', action='store_true',
                        default=False, help='Whether or not to train model')
    parser.add_argument('-d','--datasets', nargs='+',
                        default = ['heart','ionosphere', 'diabetes','breast-cancer',
                        'wine','glass','yeast','leukemia', 'abalone'], help='Datasets to use')
    parser.add_argument('-m','--methods', nargs='+',
                        default = ['shap','lime'], help='Interpretability methods to compare')

    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')

    parser.add_argument('--consistency_points', type=int, default=20,
                        help='sample size for consistency estimation')

    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='number of jobs for black-box optimization')
    ### Experimental Setting
    parser.add_argument('--skip_consistency', action='store_true',
                        help='Skip consistency experiments')
    parser.add_argument('--skip_point_lip', action='store_true',
                        help='Skip point-wise Lipschitz experiments')
    parser.add_argument('--skip_dataset_lip', action='store_true',
                        help='Skip dataset Lipschitz experiments')
    ### Paths and Saving
    parser.add_argument('--model_path', type=str,
                        default='checkpoints', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='out',
                        help='where to dump model config and epoch Results')
    parser.add_argument('--log_path', type=str, default='log',
                        help='where to dump training logs  epoch Results (and config??)')
    parser.add_argument('--summary_path', type=str, default='results/summary.csv',
                        help='where to dump model config and epoch Results')
    parser.add_argument('--overwrite', action='store_true',
                        default=False, help='Overwrite results')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode (verbose, no saving)')

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args

def main():
    args = parse_args()
    print(" ******* Remember to revert to *4 calls in (2) !!!!!!!!!")
    # DATA LOADING
    model_path, log_path, results_path = generate_dir_names(
        'uci', args, make=not args.debug)

    classif_dataset_names = args.datasets

    pbard = tqdm(classif_dataset_names, leave=False)
    for d, dataname in enumerate(pbard):
        pbard.set_description(f'Dataset: {dataname} ({d}/{len(classif_dataset_names)})')

        if dataname in check_completed_datasets(results_path) and not (args.overwrite):
            print('Results for {} found, skipping'.format(dataname))
            continue

        #if dataname in ['breast-cancer','diabetes']: continue #TODO: Find out what's wrong with this one

        (x_train, y_train), (x_test, y_test), features, classes = load_uci_data(dataname)

        classifier = train_classifier(x_train, y_train, x_test, y_test, classes)

        All_Results  = {}
        explainers   = {}

        for k in args.methods:
            if k == 'lime':
                explainers['LIME'] = lime_wrapper(classifier.predict_proba,
                                    lime_type = 'tabular',
                                    mode      = 'classification',
                                    multiclass=True,
                                    feature_names = features,
                                    class_names   = classes,
                                    train_data       = (x_train,y_train),
                                    feature_selection = 'none', # so that wwe get attribs for all features!
                                    num_samples  = 100,
                                    verbose = False)
            elif k == 'shap':
                explainers['SHAP'] = shap_wrapper(classifier.predict_proba,
                                    shap_type = 'kernel',
                                    link      = 'identity',
                                    mode      = 'classification',
                                    multiclass=True,
                                    feature_names = features,
                                    class_names   = classes,
                                    train_data       =  (x_train,y_train),
                                    nsamples  = 100,
                                    verbose = False)

            elif k == 'woe':
                explainers['WOE'] = woe_wrapper(classifier,
                                    mode      = 'classification',
                                    multiclass=True,
                                    feature_names = features,
                                    class_names   = classes,
                                    train_data    =  (x_train,y_train),
                                    nsamples  = 100,
                                    verbose = False)
            else:
                raise ValueError('Unrecognized explainer method')


        ### 1. Consistency analysis over multiple samples
        if not args.skip_consistency:
            print('**** Performing consistency estimation ****')
            Consistency_dict = {k: {} for k in explainers.keys()}
            mini_test = x_test[np.random.choice(len(x_test), args.consistency_points)]
            interventions = ['median', 'heighten', 'deflate']
            for k, expl in explainers.items():
                consist_res = expl.compute_dataset_consistency(mini_test, interventions)
                pickle.dump(consist_res, open(results_path + '/{}_consistency_{}.pkl'.format(dataname, k), "wb"))
                Consistency_dict[k] = consist_res

            All_Results['Consistency'] = Consistency_dict


        ### 2. Single example lipschitz estimate
        if not args.skip_point_lip:
            print('**** Performing lipschitz estimation for a single point ****')
            idx = 0
            print('Example index: {}'.format(idx))
            x = x_test[idx]
            fx = classifier.predict_proba(x.reshape(1,-1)).squeeze()
            Argmax_dict = {k: {} for k in explainers.keys()}
            for k, expl in explainers.items():
                print(f'Method: {k}')
                lip, argmax = expl.local_lipschitz_estimate(x, bound_type='box_std',
                                                            optim=args.optim,
                                                            eps=args.lip_eps,
                                                            n_calls=args.lip_calls,
                                                            n_jobs=args.n_jobs,
                                                            verbose=2)
                fargmax = classifier.predict_proba(argmax.reshape(1,-1)).squeeze()
                att_x = expl(x, None, show_plot=False)
                att_argmax = expl(argmax, None, show_plot=False)
                Argmax_dict[k] = {'lip': lip, 'x': x,  'fx': fx, 'att_x': att_x,
                                 'argmax': argmax, 'fargmax':fargmax, 'att_argmax': att_argmax}
                fpath = os.path.join(results_path, '{}_argmax_lip_{}').format(dataname, k)
                #lipschitz_argmax_plot(x, argmax, att, att_argmax, lip, save_path=fpath + '.pdf')
                pickle.dump(Argmax_dict[k], open(fpath+'.pkl',"wb"))
                if x_train.shape[1] < 30:
                    # Beyond 30 is hard to visualize
                    lipschitz_feature_argmax_plot(x, argmax, att_x, att_argmax, pred_x=fx, pred_y=fargmax,
                                                  feat_names = expl.feature_names,
                                                  save_path=fpath + '.pdf')
            pickle.dump(Argmax_dict, open(results_path + '/{}_argmax_lip_combined.pkl'.format(dataname), "wb"))
            All_Results['lipshitz_argmax'] = Argmax_dict


        ### 3. Local lipschitz estimate over multiple samples
        if not args.skip_dataset_lip:
            print('**** Performing lipschitz estimation over subset of dataset ****')
            LipResults = {k: {} for k in explainers.keys()}
            # Do filtering here (instead of inside estimate_d_lip so that sample is same for all methods)
            maxpoints = args.lip_points + 8
            mini_test = x_test[np.random.choice(len(x_test), maxpoints)]
            for k, expl in explainers.items():
                Lips = []
                if k in ['LIME', 'SHAP', 'WOE']:
                    Lips = expl.estimate_dataset_lipschitz(mini_test[:100],  # LIME is too slow!
                                                           n_jobs=args.n_jobs, bound_type='box_std',
                                                           eps=args.lip_eps, optim=args.optim,
                                                           n_calls=args.lip_calls, verbose=0)
                else:
                    Lips = expl.estimate_dataset_lipschitz(mini_test,
                                                           n_jobs=args.n_jobs, bound_type='box_std',
                                                           eps=args.lip_eps, optim=args.optim,
                                                           n_calls=args.lip_calls, verbose=0)
                LipResults[k]['g_lip_dataset'] = Lips
                print('Local g-Lipschitz estimate for {}: {:8.2f}'.format(k, Lips.mean()))
                pickle.dump(LipResults[k], open(results_path+'/{}_robustness_metrics_{}.pkl'.format(dataname, k), "wb"))

            All_Results['stability_blackbox'] = LipResults
            pickle.dump(LipResults, open(results_path + '/{}_robustness_metrics_combined.pkl'.format(dataname), "wb"))

        #pickle.dump(All_Results, open(results_path + '_combined_metrics.pkl'.format(dataname), "wb"))

        #df_lips = pd.DataFrame({k: LipResults[k]['g_lip_dataset'] for k in Results.keys()})
        # lipschitz_boxplot(df_lips, continuous=True,
        #                   save_path=os.path.join(results_path, 'local_glip_comparison.pdf'))


if __name__ == '__main__':
    main()
