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
from pprint import pprint
from attrdict import AttrDict
import scipy as sp
import numpy as np
from skopt import gp_minimize, gbrt_minimize
from functools import partial
from collections import defaultdict
from itertools import chain
import pickle
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm
#import multiprocessing
#from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed

from sklearn.metrics.pairwise import pairwise_distances


# LIME
try:
    import lime
    from lime import lime_tabular, lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
except:
    print('lime not installed. Robust interpret wont be able to test LIME')

# SHAP
try:
    import shap
except:
    print('shap not installed. Robust interpret wont be able to test SHAP')


# DEEP Explain
try:
    import tensorflow as tf
    from keras import backend as K
    import keras.models
    from keras.models import Sequential, Model
    from deepexplain.tensorflow import DeepExplain
    from keras.wrappers.scikit_learn import KerasClassifier
    withkeras = True
except:
    withkeras = False
    print('Tensorflow/Keras. Robust interpret wont be able to test DeepExplain models')


# UTILS
from .utils import deepexplain_plot, plot_prob_drop
from .utils import rgb2gray_converter
from .utils import topk_argmax



#   - make all wrappers uniform by passing model, not just predict_proba method,
#     since DeepX needs the full model. Can then take whatever method is required inside each class.
#   - seems like for deepexplain can choose between true and predicted class easlity. Add that option.


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)

        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def _parallel_lipschitz(wrapper, i, x, bound_type, eps, n_calls):
    if withkeras: make_keras_picklable()
    print('\n\n ***** PARALLEL : Example ' + str(i) + '********')
    print(wrapper.net.__dict__.keys())
    if 'model' in wrapper.net.__dict__.keys():
        l, _ = wrapper.local_lipschitz_estimate(
            x, eps=eps, bound_type=bound_type, n_calls=n_calls)
    else:
        l = None
    return l


class explainer_wrapper(object):
    """
        Wrapper class for interpretability methods - yields a unified callable
        for easy comparison of methods.
    """
    def __init__(self, model, mode, explainer, multiclass=False,
                 feature_names=None, class_names=None, train_data=None):
        self.mode = mode
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.multiclass = multiclass
        self.class_names = class_names
        self.x_train, self.y_train = train_data  # Necessary only to get data distrib stats

        if self.x_train is not None:
            # These are per-feature dim statistics
            print("Computing train data stats...")
            self.x_stats = AttrDict({
                'min': self.x_train.min(0),
                'max': self.x_train.max(0),
                'mean': self.x_train.mean(0),
                'median': np.median(self.x_train,axis=0),
                'std': self.x_train.std(0)
            })

    def estimate_dataset_lipschitz(self, dataset, continuous=True, eps=1,
                                   maxpoints=None, optim='gp', bound_type='box',
                                   n_jobs=1, n_calls=10, verbose=False):
        """
            Continuous and discrete space version.

        """
        if withkeras: make_keras_picklable()
        n = len(dataset)
        if maxpoints and n > maxpoints:
            dataset_filt = dataset[np.random.choice(n, maxpoints)]
        else:
            dataset_filt = dataset[:]
        if n_jobs > 1:
            Lips = Parallel(n_jobs=n_jobs, max_nbytes=1e6, verbose=verbose)(delayed(_parallel_lipschitz)(
                self, i=i, x=x, bound_type=bound_type, eps=eps, n_calls=n_calls) for i, x in enumerate(dataset_filt))
        else:
            Lips = []
            for x in dataset_filt:
                l, _ = self.local_lipschitz_estimate(x, optim=optim,
                                                     bound_type=bound_type, eps=eps, n_calls=n_calls, verbose=verbose)
                Lips.append(l)
        print(
            'Missed points: {}/{}'.format(sum(x is None for x in Lips), len(dataset_filt)))
        Lips = np.array([l for l in Lips if l])
        return Lips

    def lipschitz_ratio(self, x=None, y=None, reshape=None, minus=False):
        """
            If minus = True, returns minus this quantitiy.

            || f(x) - f(y) ||/||x - y||

        """
        # Need this ugly hack because skopt sends lists
        if type(x) is list:
            x = np.array(x)
        if type(y) is list:
            y = np.array(y)
        if reshape is not None:
            # Necessary because gpopt requires to flatten things, need to restrore expected sshape here
            x = x.reshape(reshape)
            y = y.reshape(reshape)
        multip = -1 if minus else 1
        return multip * np.linalg.norm(self(x) - self(y)) / np.linalg.norm(x - y)

    def local_lipschitz_estimate(self, x, optim='gp', eps=None, bound_type='box',
                                 clip=True, n_calls=100, n_jobs = -1, verbose=False):
        """
            Compute one-sided lipschitz estimate for explainer. Adequate for local
             Lipschitz, for global must have the two sided version. This computes:

                max_z || f(x) - f(z)|| / || x - z||

            Instead of:

                max_z1,z2 || f(z1) - f(z2)|| / || z1 - z2||

            If eps provided, does local lipzshitz in:
                - box of width 2*eps along each dimension if bound_type = 'box'
                - box of width 2*eps*va, along each dimension if bound_type = 'box_norm' (i.e. normalize so that deviation is eps % in each dim )
                - box of width 2*eps*std along each dimension if bound_type = 'box_std'

            max_z || f(x) - f(z)|| / || x - z||   , with f = theta

            clip: clip bounds to within (min, max) of dataset

        """
        # Compute bounds for optimization
        if eps is None:
            # If want to find global lipzhitz ratio maximizer - search over "all space" - use max min bounds of dataset fold of interest
            # gp can't have lower bound equal upper bound - so move them slightly appart
            lwr = self.x_stats['min'].flatten() - 1e-6
            upr = self.x_stats['max'].flatten() + 1e-6
        elif bound_type == 'box':
            lwr = (x - eps).flatten()
            upr = (x + eps).flatten()
        elif bound_type == 'box_std':
            # gp can't have lower bound equal upper bound - so set min std to 0.001
            lwr = (
                x - eps * np.maximum(self.x_stats['std'], 0.001)).flatten()
            upr = (
                x + eps * np.maximum(self.x_stats['std'], 0.001)).flatten()
        if clip:
            lwr = lwr.clip(min=self.x_stats['min'].min())
            upr = upr.clip(max=self.x_stats['max'].max())
        bounds = list(zip(*[lwr, upr]))
        if x.ndim > 2:
            # This is an image, will need to reshape
            orig_shape = x.shape
            x = x.flatten()
        else:
            orig_shape = x.shape

        # Run optimization
        if optim == 'gp':
            print('Running BlackBox Minimization with Bayesian Optimization')
            # Need minus because gp only has minimize method
            f = partial(self.lipschitz_ratio, x,
                        reshape=orig_shape, minus=True)
            res = gp_minimize(f, bounds, n_calls=n_calls,
                              verbose=verbose, n_jobs=n_jobs)
        elif optim == 'gbrt':
            print('Running BlackBox Minimization with Gradient Boosted Trees')
            f = partial(self.lipschitz_ratio, x,
                        reshape=orig_shape, minus=True)
            res = gbrt_minimize(f, bounds, n_calls=n_calls,
                                verbose=verbose, n_jobs=n_jobs)

        lip, x_opt = -res['fun'], np.array(res['x'])
        if verbose:
            print(lip, np.linalg.norm(x - x_opt))
        return lip, x_opt.reshape(orig_shape)

    def estimate_discrete_dataset_lipschitz(self, dataset, eps = None, top_k = 1,
        metric = 'euclidean', same_class = False):
        """
            For every point in dataset, find pair point y in dataset that maximizes
            Lipschitz: || f(x) - f(y) ||/||x - y||

            Args:
                - dataset: a tds obkect
                - top_k : how many to return
                - max_distance: maximum distance between points to consider (radius)
                - same_class: ignore cases where argmax g(x) != g(y), where g is the prediction model
        """
        Xs  = dataset
        n,d = Xs.shape
        Fs = self(Xs)
        Preds_prob = self.model(Xs)
        Preds_class = Preds_prob.argmax(axis=1)
        num_dists = pairwise_distances(Fs)#, metric = 'euclidean')
        den_dists = pairwise_distances(Xs, metric = metric) # Or chebyshev?
        if eps is not None:
            nonzero = np.sum((den_dists > eps))
            total   = den_dists.size
            print('Number of zero denom distances: {} ({:4.2f}%)'.format(
                total - nonzero, 100*(total-nonzero)/total))
            den_dists[den_dists > eps] = -1.0 #float('inf')
        # Same with self dists
        den_dists[den_dists==0] = -1 #float('inf')
        if same_class:

            for i in range(n):
                for j in range(n):
                    if Preds_class[i] != Preds_class[j]:
                        den_dists[i,j] = -1

        ratios = (num_dists/den_dists)
        argmaxes = {k: [] for k in range(n)}
        vals, inds = topk_argmax(ratios, top_k)
        argmaxes = {i:  [(j,v) for (j,v) in zip(inds[i,:],vals[i,:])] for i in range(n)}
        return vals.squeeze(), argmaxes

    def compute_dataset_consistency(self,  dataset, intervention_types='median'):
        """
            does compute_prob_drop for all dataset, returns stats and plots

        """
        if type(intervention_types) is str:
            intervention_types = [intervention_types]

        results = []

        for x in tqdm(dataset):
            attscores = self(x.reshape(1,-1)).squeeze()
            pred = self.model(x.reshape(1,-1)).argmax()

            empirical_lls = []
            for i,xi in enumerate(x):
                unique, counts = np.unique(self.y_train[np.isclose(xi, self.x_train[:,i], rtol = 1e-2)], return_counts=True)
                llratio = counts[np.where(unique==pred)[0][0]]
                empirical_lls.append(llratio)

            res = {'x': x, 'attscores': attscores, 'prior': (self.y_train == pred).sum()/len(self.y_train)}
            pdb.set_trace()
            for inter in intervention_types:
                Δp, p, p̂, xsub  = self.compute_prob_change(x, inter)
                if not 'p' in res: # p should be the same regardless of internention type, add once only
                    res['p'] = p
                if inter in ['median', 'deflate']:
                    # Consitent if high relevance ~ large negative p (i.e., prob of prdicted class drops)
                    corr = np.corrcoef(-Δp, attscores)[0,1]
                elif inter == 'heighten':
                    # Consistent if high relevance ~ p increases with heightening
                    corr = np.corrcoef(Δp, attscores)[0,1]


                res['xsub_'+inter]  = xsub
                res['phats_'+inter] = p̂
                res['delta_'+inter] = Δp
                res['corr_'+inter]  = corr

            results.append(res)

        # Convert from list of dicts to dict of lists
        results = {k: [d[k] for d in results] for k in results[0]}

        return results

    def compute_prob_change(self, x, intervention_type='median', scaling=0.1,
                          plot = False, save_path = None):
        """
            Computes change in probability of predicted class given change in input:

                    δ_i =  p(y|x̂_i) - p(y|x)   for every i

            The form of x̂ is determined by the 'intervention_type':
                - median: repalce x_i by median value in training data
                - heighten: make x_i more 'extreme' by heightening in the directon away
                            from median (i.e., add if > median, subtract if < median)
                            by a factor of heighten_scale*sd(x_i). E.g., heighten_scale=0.5
                            and x_i is 0.7 s.d. from than the median, it will now be
                            1.2 sd from the median (in same direction)

            Warning: this only works for SENN if concepts are inputs!!!!
            In that case, must use the compute prob I have in SENN class.
        """
        fx = self.model(x.reshape(1,-1))
        pred_class = fx.argmax()
        p = fx.squeeze()[pred_class]
        p̂s = []
        deltas = []
        subs_vals = []

        for i in tqdm(range(x.shape[0])):
            x̂ = x.copy()
            if intervention_type == 'median':
                x̂[i] = self.x_stats.median[i]
            elif intervention_type == 'heighten':
                direction = np.sign(x[i] - self.x_stats.median[i])
                x̂[i] = x[i] + direction*scaling*self.x_stats.std[i]
            elif intervention_type == 'deflate':
                direction = -np.sign(x[i] - self.x_stats.median[i])
                x̂[i] = x[i] + direction*scaling*self.x_stats.std[i]
            subs_vals.append(x̂[i])
            fx̂ = self.model(x̂.reshape(1,-1))
            delta_i = (fx̂-fx).squeeze()[pred_class]
            p̂s.append(fx̂.squeeze()[pred_class])
            deltas.append(delta_i)
        deltas = np.array(deltas)
        p̂s     = np.array(p̂s)
        subs_vals = np.array(subs_vals)
        # if plot:
        #     plot_prob_drop(attributions, deltas, save_path = save_path)
        return deltas, p, p̂s, subs_vals


    def plot_image_attributions_(self, x, attributions):
        """
            x: either (n x d) if features or (n x d1 x d2) if gray image or (n x d1 x d2 x c) if color
        """
        # for different types of data _display_attribs_image , etc
        n_cols = 4
        n_rows = max(int(len(attributions) / 2), 1)
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
        for i, a in enumerate(attributions):
            row, col = divmod(i, 2)
            if  (x.ndim == 3) or (x.shape[3] == 1):
                # Means single channel, like MNIST, plot in gray
                # I had this for mnist, worked:
                deepexplain_plot(x[i].reshape(
                    x.shape[1], x.shape[2]), cmap='Greys', axis=axes[row, col * 2]).set_title('Original')
                deepexplain_plot(a.reshape(
                    x.shape[1], x.shape[2]), xi=x[i], axis=axes[row, col * 2 + 1]).set_title('Attributions')
            else:
                ax = axes[row, col * 2]
                xi = x[i]
                xi = (xi - np.min(xi))
                xi /= np.max(xi)
                ax.imshow(xi)
                ax.set_title('Original')
                ax.axis('off')
                deepexplain_plot(a, xi=x[i], axis=axes[row, col * 2 + 1],
                                 dilation=.5, percentile=99, alpha=.2).set_title('Attributions')
                #deepexplain_plot(a, xi = x[i], axis=axes[row,col*2+1],dilation=.5, percentile=99, alpha=.2).set_title('Attributions')
        plt.show()


class shap_wrapper(explainer_wrapper):
    """
        Wrapper around SHAP explanation framework from shap github package by the authors
    """

    def __init__(self, model, shap_type, link, mode, multiclass=False, feature_names=None,
                 class_names=None, train_data=None, num_features=None, categorical_features=None,
                 nsamples=100, verbose=False):
        print('Initializing {} SHAP explainer wrapper'.format(shap_type))
        super().__init__(model, mode, None, multiclass,
                         feature_names, class_names, train_data)
        if shap_type == 'kernel':
            explainer = shap.KernelExplainer(model, self.x_train, link=link)

        self.explainer = explainer
        self.nsamples = nsamples

    def __call__(self, x, y=None,  x_raw=None, return_dict=False, show_plot=False):
        """
            y only needs to be specified in the multiclass case. In that case,
            it's the class to be explained (typically, one would take y to be
            either the predicted class or the true class). If it's a single value,
            same class explained for all inputs
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y is None:
            # Explain predicted class
            y = np.argmax(self.model(x).reshape(x.shape[0],len(self.class_names)), axis = 1)
        elif y is not None and y.ndim > 1:
            # Means y is given in one-hot format
            y = np.argwhere(y)[:, 1]
        elif (type(y) is int or y.ndim == 0) and (x.ndim == 1):
            # Single example
            #y = np.array([y]).reshape(1,1)
            y = [y]
            x = x.reshape(1, -1)
        elif (type(y) is int or y.ndim == 0):
            # multiple examples, same class to be explained in all of them
            y = [y] * x.shape[0]

        assert x.shape[0] == len(
            y), "x.shape = {}, len(y) = {}".format(x.shape, len(y))

        exp_classes = self.explainer.shap_values(x, nsamples=self.nsamples, verbose = False)
        # print(exp_classes)
        # if self.multiclass and type(y) is int:
        #     exp = exp_classes[y] # explanation of desired class for multiclass case
        if self.multiclass:
            exp = np.array([exp_classes[y[i]][i]
                            for i in range(len(exp_classes[0]))])
        else:
            exp = exp_classes

        if x.shape[0] == 1:
            # Squeeze if single prediction
            exp = exp[0]

        self.explanation = exp
        exp_dict = dict(zip(self.feature_names + ['bias'], exp.T.tolist()))
        vals = np.array([exp_dict[feat]
                         for feat in self.feature_names if feat in exp_dict.keys()]).T
        if not return_dict:
            return vals
        else:
            return exp_dict


class lime_wrapper(explainer_wrapper):
    """
        Wrapper around LIME explanation framework from lime github package by the authors
    """

    def __init__(self, model, lime_type, mode, multiclass=False, feature_names=None, num_samples=100,
                 class_names=None, train_data=None, feature_selection = 'auto', num_features=None, categorical_features=None, channels=3,
                 segmenter=None, verbose=False):
        print('Initializing {} LIME explainer wrapper'.format(lime_type))
        if lime_type == 'tabular':
            self.channels = channels
            explainer = lime.lime_tabular.LimeTabularExplainer(
                train_data[0], feature_names=feature_names, class_names=class_names,
                discretize_continuous=False, categorical_features=categorical_features,
                verbose=verbose, mode=mode, feature_selection=feature_selection)
        elif lime_type == 'image':
            self.channels = channels
            explainer = lime.lime_image.LimeImageExplainer(verbose=verbose)
            if self.channels == 1:
                # LIME-image needs to work with 3 channel images. Hacky way to make prediction
                # model accept 3channels if it's only desdgen for 1
                # Need minus because gp only has minimize method
                model = partial(self.predict_fn_rgb2gray, model)

        super().__init__(model, mode, explainer, multiclass,
                         feature_names, class_names, train_data)
        self.lime_type = lime_type
        #self.explainer = explainer
        self.num_features = num_features if num_features else len(
            self.feature_names)
        self.num_samples = num_samples
        self.segmenter = segmenter

    def predict_fn_rgb2gray(self, model, input):
        return model(rgb2gray_converter(input))

    def extract_att_tabular(self, exp, y):
        # """ Method for extracting a numpy array from lime explanation objects"""
        # if x.ndim == 1:
        # Single explanation
        if y is None or (type(y) is list and y[0] is None):
            #exp_dict = dict(exp.as_list()) # SOMETIMES BREAKS
            exp_dict  = dict(exp.as_list(exp.top_labels[0])) # Hacky but works
        elif self.multiclass:
            exp_dict = dict(exp.as_list(label=y))
        else:
            exp_dict = dict(exp.as_list())
        #exp_dict = dict(exp.as_list())
        # FOR NOW IGNORE DISCRETE - THEY're
        vals = np.array([exp_dict[feat]
                         for feat in self.feature_names if feat in exp_dict.keys()])
        return vals

    def extract_att_image(self, exp, y=None):

        if y is None or (type(y) in [list,tuple] and y[0] is None):
            y = exp.top_labels[0]
        scores = self.get_img_weight_matrix_(exp,
                                             y, num_features=self.num_features,
                                             min_weight=0)
        return scores

    def get_img_weight_matrix_(self, imgexp, label,
                               num_features=5, min_weight=0.):
        """ Based on lime's get_image_and_mask method in ImageExplanation class,
            modified to just return matrix of weights.

        Args:
            label: label to explain
            num_features: number of superpixels to include in explanation
            min_weight: TODO
        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in imgexp.local_exp:
            pdb.set_trace()
            raise KeyError('Label not in explanation')
        segments = imgexp.segments
        exp = imgexp.local_exp[label]
        scores = np.zeros((imgexp.image.shape[:2]))
        for f, w in exp[:num_features]:
            if np.abs(w) < min_weight:
                continue
            scores[segments == f] = w
        return scores

    def __call__(self, x, y=None, x_raw=None,  return_dict=False, show_plot=False):
        # the only thing that will change is labels and toplabels
        # if y is None:
        #     # No target class provided - use predicted
        #     y = self.model(x).argmax(1)
        #
        if (self.lime_type == 'tabular' and x.ndim == 1) or \
           (self.lime_type == 'image' and x.ndim == 3):
            if not self.multiclass:
                labs, top_labs = [(1,)], None  # Explain the "only" class
            elif y is None:
                labs, top_labs = None, 1      # Explain only most likely predicted class
            else:
                labs, top_labs = (y,), None   # Explains class y
            if self.lime_type == 'image':
                if x.shape[2] == 1:
                    # This is a grayscale image, lime needs to receive 2d in this case
                    x = x.reshape(x.shape[0], x.shape[1])
                exp = self.explainer.explain_instance(x, self.model,
                                                      labels=labs,
                                                      top_labels=top_labs,
                                                      num_features=self.num_features,
                                                      num_samples=self.num_samples,
                                                      segmentation_fn=self.segmenter
                                                      )
                attributions = self.extract_att_image(exp, y)
            elif self.lime_type == 'tabular':
                exp = self.explainer.explain_instance(x, self.model,
                                                      labels=labs,
                                                      top_labels=top_labs,
                                                      num_features=self.num_features,
                                                      num_samples=self.num_samples,
                                                      )
                attributions = self.extract_att_tabular(exp, y)
            self.explanation = exp
        else:
            # There's multiple examples to explain
            N = int(x.shape[0])
            if not self.multiclass:
                labs = [(1,)] * N
                top_labs = 1
            elif y is None:
                # Irrelevant, will be ignored with top_labs provided
                labs = [(None,)] * N
                top_labs = 1
            else:
                top_labs = None
                labs = [(y[i],) for i in range(N)]

            if self.lime_type == 'image':
                if x.shape[3] == 1:
                    # This is a grayscale image, lime needs to receive 2d in this case
                    x = np.squeeze(x)  # x.reshape(-1,-1,-1)
                exp = [
                    self.explainer.explain_instance(x[i, :],
                                                    self.model,
                                                    labels=labs[i], top_labels=top_labs,
                                                    num_features=self.num_features,
                                                    num_samples=self.num_samples,
                                                    segmentation_fn=self.segmenter
                                                    )
                    for i in range(N)
                ]
            elif self.lime_type == 'tabular':
                exp = [
                    self.explainer.explain_instance(x[i, :],
                                                    self.model,
                                                    labels=labs[i], top_labels=top_labs,
                                                    num_features=self.num_features,
                                                    num_samples=self.num_samples,
                                                    )
                    for i in range(N)
                ]
            self.explanation = exp
            if self.lime_type == 'tabular':
                attributions = [self.extract_att_tabular(
                    exp[i], labs[i][0]) for i in range(len(exp))]
            elif self.lime_type == 'image':
                attributions = [self.extract_att_image(
                    exp[i], labs[i]) for i in range(len(exp))]
            #vals = np.stack(vals, axis = 1)
            attributions = np.stack(attributions, axis=0)

        if show_plot and self.lime_type == 'image':
            x_plot = x_raw if (x_raw is not None) else x
            self.plot_image_attributions_(x_plot, attributions)

        # Hacky. For RBG images, all other methods return one attrib per channelself, so do same here
        if self.channels == 3 and self.lime_type == 'image':
            assert x.shape[-1] == 3
            attributions = np.tile(attributions[:,:,:,np.newaxis], (1,1,1,3))

        if attributions.ndim > 1:
            # WHY DO WE NEED THIS? SEEMSS LIKE ELSE version is the way to unifoirmize one vs multiple examples
            # Was this needed for images?
            attributions = attributions.reshape(attributions.shape[0], -1)
        # else:
        #     attributions = attributions.reshape(1, attributions.shape[0])
        if not return_dict:
            return attributions
        else:
            return exp_dict


class deepexplain_wrapper(explainer_wrapper):
    """
        Wrapper around deep explain explanation framework

        Explainer type is one of:
        # Gradient-based
            - 'saliency'    (Saliency maps)
            - 'grad*input'  (Gradient*Input)
            - 'intgrad'     (Integrated Gradients)
            - 'elrp'        (Epsilon-LRP)
            - 'deeplift'    (Deep LIFT)
        # Perturbation-based
            - 'occlusion'   (Occlusion) Has window_shape arg
    """
    # def __init__(self, keras_wrapper, explainer, multiclass, feat_names, class_to_explain, train_data, nsamples = 100):
    #     super().__init__(keras_wrapper.predict, None, multiclass, feat_names, train_data)

    def __init__(self, model, method_type, mode, multiclass=False, feature_names=None,
                 class_names=None, train_data=None, num_features=None, categorical_features=None,
                 nsamples=100, verbose=False):
        print('Initializing "{}" DeepExplain wrapper'.format(method_type))
        #self.num_feat = num_feat if num_feat else len(self.feature_names)
        #self.keras_wrapper   = model
        if type(model) is KerasClassifier:
            self.net = model.model
        elif type(model) in [Model, Sequential]:
            self.net = model
        super().__init__(self.net.predict, mode, None,
                         multiclass, feature_names, class_names, train_data)
        self.method_type = method_type
        self.nsamples = nsamples
        self.verbose = verbose
        #self.c = class_to_explain

    def __call__(self, x, y=None, return_dict=False, x_raw=None, show_plot=False):
        """
            x_raw: if provided, will plot this instead of x (useful for images that have been processed)

        """
        if x.ndim == 3:  # Means x is image and single example passed. Batchify.
            x = x.reshape(1, *x.shape)

        if self.multiclass:
            print("Do wsomething")

        # NOTE: Seems Deepexplain by default explains true class in multiclass case
        with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
            # Need to reconstruct the graph in DeepExplain context, using the same weights.
            # With Keras this is very easy:
            # 1. Get the input tensor to the original model
            input_tensor = self.net.layers[0].input

            # 2. We now target the output of the last dense layer (pre-softmax)
            # To do so, create a new model sharing the same layers untill the last dense (index -2)
            fModel = Model(inputs=input_tensor,
                           outputs=self.net.layers[-2].output)
            target_tensor = fModel(input_tensor)
            if y is None:
                # Will compute attribution w.r.t. predicted class
                expl_target_class = tf.reduce_max(target_tensor, 1)
            else:
                # Will compute attribution w.r.t. to provided classes
                expl_target_class = target_tensor * y

            attributions = de.explain(
                self.method_type, expl_target_class, input_tensor, x)

        if show_plot:
            x_plot = x_raw if (x_raw is not None) else x
            self.plot_image_attributions_(x_plot, attributions)

        self.explanation = attributions
        vals = attributions.reshape(attributions.shape[0], -1)
        if not return_dict:
            return vals
        else:
            exp_dict = dict(zip(self.feature_names, vals))
            return exp_dict

class gsenn_wrapper(explainer_wrapper):
    """
        Wrapper around gsenn
    """
    # def __init__(self, keras_wrapper, explainer, multiclass, feat_names, class_to_explain, train_data, nsamples = 100):
    #     super().__init__(keras_wrapper.predict, None, multiclass, feat_names, train_data)
    def __init__(self, model, mode,input_type, multiclass=False, feature_names=None,
                 class_names=None, train_data=None, num_features=None, categorical_features=None,
                 nsamples=100, skip_bias = True, verbose=False):
        print('Initializing GSENN wrapper.')
        #self.num_feat = num_feat if num_feat else len(self.feature_names)
        #self.keras_wrapper   = model
        self.input_type = input_type
        self.net = model
        self.skip_bias = skip_bias # If we added bias term in SENN, remove from attributions
        # if type(model) is KerasClassifier:
        #     self.net = model.model
        # elif type(model) in [Model, Sequential]:
        #     self.net = model
        super().__init__(self.net.predict_proba, mode, None,
                         multiclass, feature_names, class_names, None)

        # dataloader or Dataset that does not involve iterating
        stack = []
        for input,_ in train_data:
            stack.append(input.squeeze().numpy())

        transformed_dataset = np.concatenate(stack)
        if train_data is not None:
            print("Computing train data stats...")
            self.x_stats = {
                'min': np.min(transformed_dataset,0),
                'max': np.max(transformed_dataset,0),
                'mean': np.mean(transformed_dataset,0),
                'std': np.std(transformed_dataset,0),
            }

        self.verbose = verbose


    def __call__(self, x, y=None, return_dict=False, x_raw=None, show_plot=False):
        """
            x_raw: if provided, will plot this instead of x (useful for images that have been processed)

        """
        import torch
        from torch import Tensor, from_numpy
        from torch.autograd import Variable

        if type(x) is np.ndarray:
            x_t = from_numpy(x).float()
        elif type(x) is Tensor:
            x_t = x.clone()
        else:
            print(type(x))
            raise ValueError("Unrecognized data type")

        if x_t.dim() == 1:
            x_t = x_t.view(1,-1)
        elif x_t.dim() == 2 and self.input_type == 'image':
            # Single image, gray. Batchify with channels first
            x_t = x_t.view(1,1,x_t.shape[0],x_t.shape[1])
        elif x_t.dim() == 3:  # Means x is image and single example passed. Batchify.
            x_t = x_t.view(1, x_t.shape[0], x_t.shape[1], x_t.shape[2])

        if self.input_type == 'image' and (x_t.shape[1] in [1,3]):
            channels_first = True
        elif self.input_type == 'image':
            channels_first = False
            x_t = x_t.transpose(1,3).transpose(2,3)

        if self.multiclass:
            pass

        x_t = Variable(x_t, volatile = True)
        pred = self.net(x_t)

        attrib_mat = self.net.thetas.data.cpu()

        nx, natt, nclass = attrib_mat.shape

        if y is None:
            # Will compute attribution w.r.t. predicted class
            vals, argmaxs = torch.max(pred.data, -1)
            attributions = attrib_mat.gather(2,argmaxs.view(-1,1).unsqueeze(2).repeat(1,natt,nclass))[:,:,0].numpy()
            #attributions = attrib_mat[:,:,argmaxs].squeeze(-1).numpy()
            #attributions = attributions.reshape()
        else:
            # Will compute attribution w.r.t. to provided classes
            expl_target_class = target_tensor * y


        if self.skip_bias and getattr(self.net.conceptizer, "add_bias", None):
            attributions = attributions[...,:-1]
            #attributions = torch.index_select(attributions,-1,torch.LongTensor(range(attributions.shape[-1]-1)))

        if show_plot:
            if self.input_type == 'image':
                x_plot = x_raw if (x_raw is not None) else x
                if x_raw is None and channels_first:
                    x_plot = x_plot.transpose(1,3).transpose(1,2)
                if not type(x_plot) is np.ndarray:
                    x_plot = x_plot.numpy().squeeze()

                #x_plot = x_plot.squeeze()
                self.plot_image_attributions_(x_plot, attributions)
            else: #if self.input_type == 'feature':
                exp_dict_y = dict(zip(feat_names, att_y))
                _ = plot_dependencies(exp_dict_y, x = y, sort_rows = False, scale_values = False, ax = ax[1,1],
                                        show_table = True, digits = digits, ax_table = ax[1,0], title = 'Explanation')

        self.explanation = attributions
        vals = attributions.reshape(attributions.shape[0], -1)

        if not return_dict:
            return vals
        else:
            exp_dict = dict(zip(self.feature_names, vals))
            return exp_dict


import sys
sys.path.append('../../workspace/interpretwoe/')
from interpretwoe.explainers import WoEExplainer
from interpretwoe.woe import woe_gaussian

class woe_wrapper(explainer_wrapper):
    def __init__(self, model, mode, woe_explainer=None,
                 multiclass=False, feature_names=None,
                 class_names=None, train_data=None, num_features=None,
                 expl_units='features',
                 featgroup_idxs = None,
                 featgroup_names= None,
                 nsamples=100, verbose=False):


        #pdb.set_trace()
        if woe_explainer is None:
            assert expl_units == 'features' or not (featgroup_idxs is None)
            print('Initializing WOE explainer wrapper')
            woe_model = woe_gaussian(model, train_data[0], classes = range(len(class_names)), cond_type='nb')

            # if expl_units == 'features':
            #     featgroup_idxs = np.arange(len(feature_names))
            #     featgroup_names = feature_names

            # Instantiate woe-based explainer
            woe_explainer = WoEExplainer(model, woe_model,
                                  total_woe_correction=False,
                                  classes=class_names, features=feature_names,
                                  X=train_data[0], Y=None,#Y_train,
                                  featgroup_idxs = featgroup_idxs,
                                  featgroup_names = featgroup_names)


        self.explainer = woe_explainer
        self.units = expl_units
        super().__init__(model.predict_proba, mode, woe_explainer, multiclass,
                         feature_names, class_names, train_data)

    def __call__(self, x, y=None,  x_raw=None, return_dict=False, show_plot=False):
        """
            y only needs to be specified in the multiclass case. In that case,
            it's the class to be explained (typically, one would take y to be
            either the predicted class or the true class). If it's a single value,
            same class explained for all inputs
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y is None:
            y = np.argmax(self.model(x))
        y = y.reshape(1,)

        exp = self.explainer.explain(x, y, units=self.units,
                                   print=False, plot=False)

        self.explanation = exp

        exp_dict = dict(zip(self.feature_names + ['bias'], exp.attwoes.tolist() + [exp.prior_lodds]))
        vals = np.array([exp_dict[feat]
                         for feat in self.feature_names if feat in exp_dict.keys()]).T
        if not return_dict:
            return vals
        else:
            return exp_dict

        # if x.ndim == 1:
        #     x = x.reshape(1, -1)
        # if y is None:
        #     # Explain predicted class
        #     y = np.argmax(self.model(x).reshape(x.shape[0],len(self.class_names)), axis = 1)
        # elif y is not None and y.ndim > 1:
        #     # Means y is given in one-hot format
        #     y = np.argwhere(y)[:, 1]
        # elif (type(y) is int or y.ndim == 0) and (x.ndim == 1):
        #     # Single example
        #     #y = np.array([y]).reshape(1,1)
        #     y = [y]
        #     x = x.reshape(1, -1)
        # elif (type(y) is int or y.ndim == 0):
        #     # multiple examples, same class to be explained in all of them
        #     y = [y] * x.shape[0]
        #
        # assert x.shape[0] == len(
        #     y), "x.shape = {}, len(y) = {}".format(x.shape, len(y))
        #
        # exp_classes = self.explainer.shap_values(x, nsamples=self.nsamples, verbose = False)
        # # print(exp_classes)
        # # if self.multiclass and type(y) is int:
        # #     exp = exp_classes[y] # explanation of desired class for multiclass case
        # if self.multiclass:
        #     exp = np.array([exp_classes[y[i]][i]
        #                     for i in range(len(exp_classes[0]))])
        # else:
        #     exp = exp_classes
        #
        # if x.shape[0] == 1:
        #     # Squeeze if single prediction
        #     exp = exp[0]
        #
        # self.explanation = exp
        # exp_dict = dict(zip(self.feature_names + ['bias'], exp.T.tolist()))
        # vals = np.array([exp_dict[feat]
        #                  for feat in self.feature_names if feat in exp_dict.keys()]).T
        # if not return_dict:
        #     return vals
        # else:
        #     return exp_dict
