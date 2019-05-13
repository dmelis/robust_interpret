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
# Necessary for parallelism in joblib + keras
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
# 0: all logs shown, 1: filter out INFO logs 2: filter out WARNING logs, and 3 to additionally filter out ERROR log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tempfile
import sys
sys.path.insert(0, os.path.abspath('..'))
import warnings
from functools import partial
import numpy as np
import matplotlib
    # Hack to be able to save plots in janis
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import pickle
import argparse
import pandas as pd

# Keras
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr  # To filter out "Using tensorflow backend."
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain

# Local imports
# from os.path import dirname, realpath
# sys.path.append(os.path.join(dirname(realpath(__file__)), 'codebase/'))

import robust_interpret.utils
import robust_interpret.explainers as explainers

from robust_interpret.explainers import deepexplain_wrapper, lime_wrapper
from lime.wrappers.scikit_image import SegmentationAlgorithm

from robust_interpret.utils import deepexplain_plot, lipschitz_boxplot, lipschitz_argmax_plot
from robust_interpret.utils import generate_dir_names  # plot_theta_stability,
from robust_interpret.utils import plot_attribution_stability


def load_mnist_data(batch_size=128):
    # Data Processing
    num_classes = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = (x_train - 0.5) * 2
    x_test = (x_test - 0.5) * 2
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train_hot = keras.utils.to_categorical(y_train, num_classes)
    y_test_hot = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), y_train_hot, y_test_hot

def build_by_loading(self, model_path):
    model = load_model(model_path)
    return model

def create_model(activation="relu"):
    input_shape = (28, 28, 1)
    num_classes = 10
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation=activation,
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # ^ IMPORTANT: notice that the final softmax must be in its own layer
    # if we want to target pre-softmax units
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def noise_stability_plots(model, dataset, cuda, save_path):
    # find one example of each digit:
    examples = {}
    i = 0
    while (not len(examples.keys()) == 10) and (i < len(dataset)):
        if dataset[i][1] not in examples:
            examples[dataset[i][1]] = dataset[i][0].view(1, 1, 28, 28)
        i += 1

    for i in range(10):
        x = Variable(examples[i], volatile=True)
        if cuda:
            x = x.cuda()
        plot_theta_stability(model, x, noise_level=0.5,
                             save_path=save_path + '/noise_stability_{}.pdf'.format(i))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Interpteratbility robustness evaluation on MNIST')

    # #setup
    parser.add_argument('--train', action='store_true',
                        default=False, help='Whether or not to train model')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=1.0,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    # #paths
    parser.add_argument('--model_path', type=str,
                        default='models', help='where to save the snapshot')
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
    explainers.make_keras_picklable()
    args.nclasses = 10
    args.theta_dim = args.nclasses
    warnings.simplefilter("ignore", UserWarning)

    # DATA LOADING
    model_path, log_path, results_path = generate_dir_names(
        'mnist', args, make=not args.debug)
    (x_train, y_train), (x_test, y_test), y_train_hot, y_test_hot = load_mnist_data()

    # TRAIN/LOAD MODEL
    if not args.train:
        #build_fn = partial(build_by_loading, model_path)
        model = load_model(model_path)
    else:
        #build_fn = create_model
        # , epochs=10, batch_size=10, verbose=0)
        classifier = KerasClassifier(build_fn=create_model, verbose=1)

        # Alternative approach - grid search
        param_grid = {
            "activation": ["relu", "sigmoid"],
            "verbose": [1],
            "batch_size": [32],
            "epochs": [3]
        }
        random_search = GridSearchCV(classifier,
                                     param_grid=param_grid, verbose=1)  # , scoring="roc_auc")
        random_search.fit(x_train, y_train_hot, verbose=1)
        classifier = random_search.best_estimator_
        classifier.sk_params['verbose'] = 0
        model = classifier.model
        model.save(model_path)

    # CHECK MODEL QUALITY
    loss, acc = model.evaluate(x_test, y_test_hot)
    print('loss = {:8.2f}, acc={:8.2f}%'.format(loss, acc * 100))

    # Instantiate Interpreters

    pixel_names = ['P-{}.{}'.format(i, j)
                   for i in range(28) for j in range(28)]
    digit_names = [int(a) for a in range(10)]

    explainer_dict = {}
    deeplift_methods = {
        'Saliency': 'saliency',
        'Grad*Input': 'grad*input',
        'Int.Grad.': 'intgrad',
        'e-LRP': 'elrp',
        'Occlusion': 'occlusion'
    }

    for k, v in deeplift_methods.items():
        explainer_dict[k] = deepexplain_wrapper(model,
                                                method_type=v,
                                                mode='classification',
                                                feature_names=pixel_names,
                                                class_names=digit_names,
                                                train_data=x_train,
                                                verbose=False)

    segmenter = SegmentationAlgorithm(
        'quickshift', kernel_size=1, max_dist=200, ratio=0.1)
    LIME = lime_wrapper(model.predict,  # Need predict_proba for attribution
                        lime_type='image',
                        mode='classification',
                        multiclass=True,
                        feature_names=pixel_names,
                        class_names=digit_names,
                        train_data=x_train,
                        num_samples=1000,
                        num_features=10,
                        channels=1,
                        segmenter=segmenter,
                        verbose=False)

    explainer_dict['LIME'] = LIME

    # Eval Model
    Results = {k: {} for k in explainer_dict.keys()}

    ### 1. Qualitative Example - Variation under gaussian noise
    print('**** Performing qualitative evaluation of robustness with Gaussian noise ****')
    x = x_test[0]
    plot_attribution_stability(model, explainer_dict, x, noise_level=.37,
                               layout='horizontal', samples=3, show_proba=True,
                               save_path=os.path.join(results_path, 'gaussian_perturb_comparison.pdf'))

    ### 2. Single example lipschitz estimate
    print('**** Performing lipschitz estimation for a single point ****')
    idx = 0
    print('Example index: {}'.format(idx))
    x = x_test[idx]
    Argmax_dict = {k: {} for k in explainer_dict.keys()}
    for k, expl in explainer_dict.items():
        print(k)
        continue
        lip, argmax = expl.local_lipschitz_estimate(x, bound_type='box_std',
                                                    optim=args.optim,
                                                    eps=args.lip_eps,
                                                    n_calls=4*args.lip_calls,
                                                    verbose=2)
        Results[k]['lip_argmax'] = (x, argmax, lip)
        # .reshape(inputs.shape[0], inputs.shape[1], -1)
        att = expl(x, None, show_plot=False)
        # .reshape(inputs.shape[0], inputs.shape[1], -1)
        att_argmax = expl(argmax, None, show_plot=False)
        Argmax_dict[k] = {'lip': lip, 'argmax': argmax, 'x': x}
        fpath = os.path.join(results_path, 'argmax_lip_{}.pdf'.format(k))
        lipschitz_argmax_plot(x, argmax, att, att_argmax, lip, save_path=fpath)
        pickle.dump(Argmax_dict[k], open(
            results_path + '/argmax_lip_{}.pkl'.format(k), "wb"))

    #pickle.dump(Argmax_dict, open(results_path + '/argmax_lip_combined.pkl', "wb"))

    ### 3. Local lipschitz estimate over multiple samples
    print('**** Performing lipschitz estimation over subset of dataset ****')
    # Do filtering here (instead of inside estimate_d_lip so that sample is same for all methods)
    maxpoints = args.lip_points + 8
    mini_test = x_test[np.random.choice(len(x_test), maxpoints)]
    for k, expl in explainer_dict.items():
        Lips = []
        if k == 'LIME':
            Lips = expl.estimate_dataset_lipschitz(mini_test[:40],  # LIME is too slow!
                                                   n_jobs=1, bound_type='box_std',
                                                   eps=args.lip_eps, optim=args.optim,
                                                   n_calls=args.lip_calls, verbose=0)
        else:
            Lips = expl.estimate_dataset_lipschitz(mini_test,
                                                   n_jobs=-1, bound_type='box_std',
                                                   eps=args.lip_eps, optim=args.optim,
                                                   n_calls=args.lip_calls, verbose=0)
        Results[k]['g_lip_dataset'] = Lips
        print('Local g-Lipschitz estimate for {}: {:8.2f}'.format(k, Lips.mean()))
        pickle.dump(Results[k], open(results_path+'/robustness_metrics_{}.pkl'.format(k), "wb"))

    pickle.dump(Results, open(results_path + '/robustness_metrics_combined.pkl', "wb"))

    df_lips = pd.DataFrame({k: Results[k]['g_lip_dataset'] for k in Results.keys()})
    lipschitz_boxplot(df_lips, continuous=True,
                      save_path=os.path.join(results_path, 'local_glip_comparison.pdf'))


    print('Done!')

if __name__ == '__main__':
    main()
