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

import os, sys
import argparse
import numpy as np
import pdb

import matplotlib
    # Hack to be able to save plots in janis
    matplotlib.use('Agg')

import tensorflow as tf
from keras.preprocessing import image
from keras.applications import resnet50 as resnet

from lime.wrappers.scikit_image import SegmentationAlgorithm

# Local imports
from os.path import dirname, realpath
sys.path.append(os.path.join(dirname(realpath(__file__)),'codebase/'))
from explainers import deepexplain_wrapper, lime_wrapper
from utils import generate_dir_names, lipschitz_argmax_plot #plot_theta_stability,


def load_images_for_resnet():
    images     = np.zeros((4, 224, 224, 3))
    raw_images = np.zeros((4, 224, 224, 3))
    filenames = []
    idx = 0
    for filepath in tf.gfile.Glob(os.path.join('../data/images', '*.png')):
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        raw_images[idx,:,:,:] = x
        x = resnet.preprocess_input(x)
        images[idx,:,:,:] = x
        filenames.append(os.path.basename(filepath))
        idx += 1
    return filenames, images, raw_images

def inverse_preprocess_image(x):
    mean = [103.939, 116.779, 123.68]
    std = None
    if std is not None:
        x[..., 0] *= std[0]
        x[..., 1] *= std[1]
        x[..., 2] *= std[2]
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    return x

def parse_args():
    parser = argparse.ArgumentParser(description='Interpteratbility robustness evaluation on Images')

    # #setup
    #parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--lip_calls', type=int, default=10, help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=1.0, help='eps for Lipschitz estimation')

    # #paths
    parser.add_argument('--model_path', type=str, default='checkpoints', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='out', help='where to dump model config and epoch Results')
    parser.add_argument('--log_path', type=str, default='log', help='where to dump training logs  epoch Results (and config??)')
    parser.add_argument('--summary_path', type=str, default='results/summary.csv', help='where to dump model config and epoch Results')

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


def main():
    args = parse_args()

    ### DATA LOADING
    model_path, log_path, results_path = generate_dir_names('images', args, make =True)
    files, x_test, x_raw = load_images_for_resnet()
    model = resnet.ResNet50(weights='imagenet')

    ### Instantiate Interpreters

    pixel_names = ['P-{}.{}'.format(i,j) for i in range(28) for j in range(28)]
    class_names = [a[0][1] for a in resnet.decode_predictions(np.identity(1000), top=1)]

    explainer_dict = {}
    deeplift_methods = {
        'Saliency': 'saliency',
        'Grad*Input': 'grad*input',
        'Int.Grad.': 'intgrad',
        'e-LRP': 'elrp',
        'Occlusion': 'occlusion'
    }

    for k,v in deeplift_methods.items():
        explainer_dict[k] = deepexplain_wrapper(model,
                            method_type = v,
                            mode      = 'classification',
                            feature_names = pixel_names,
                            class_names   = class_names,
                            train_data       = None,
                            verbose = False)

    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.1)
    LIME = lime_wrapper(model.predict, # Need predict_proba for attribution
                        lime_type = 'image',
                        mode      = 'classification',
                        multiclass = True,
                        feature_names = pixel_names,
                        class_names   = class_names,
                        train_data       = None,
                        num_samples     = 1000,
                        num_features    = 10,
                        channels        = 1,
                        segmenter = segmenter,
                        verbose = False)

    explainer_dict['LIME'] = LIME

    Results = {k: {} for k in explainer_dict.keys()}

    # 2. Single example lipschitz estimate
    print('**** Performing lipschitz estimation for a single point ****')
    idx = 0
    print('Example index: {}'.format(idx))
    x = x_test[idx]

    for k, expl in explainer_dict.items():
        print(k)
        lip, argmax = expl.local_lipschitz_estimate(x, bound_type = 'box', optim = 'gp',
                        eps = args.lip_eps, n_calls = args.lip_calls, verbose = True)
        Results[k]['lip_argmax'] = (x, argmax, lip)
        att        = expl(x, None, show_plot = False)#.reshape(inputs.shape[0], inputs.shape[1], -1)
        att_argmax = expl(argmax, None, show_plot = False)#.reshape(inputs.shape[0], inputs.shape[1], -1)
        fpath = os.path.join(results_path, k + '_argmax_lip.pdf')
        lipschitz_argmax_plot(x_raw[idx], inverse_preprocess_image(argmax),
                              att, att_argmax, lip, save_path = fpath)
        pickle.dump(Results[k], open(results_path + '/robustness_metrics_{}.pkl'.format(k), "wb"))

    pickle.dump(Results, open(results_path + '/robustness_metrics_combined.pkl', "wb"))
    print('Done!')

if __name__ == '__main__':
    main()
