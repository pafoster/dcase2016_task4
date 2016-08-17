#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2016::Domestic Audio Tagging / Baseline System
# Copyright (C) 2015 Toni Heittola (toni.heittola@tut.fi) / TUT
# Copyright (C) 2016 Peter Foster (p.a.foster@qmul.ac.uk) / QMUL
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from src.ui import *
from src.general import *
from src.features import *
from src.dataset import *
from src.dataset_chimehome import *
from src.evaluation import *
from src.eer import *

import yaml
import numpy
import cPickle as pickle
import csv
import warnings
import argparse
import textwrap
import math
import librosa

import pdb

from sklearn import mixture, metrics

from IPython import embed

__version_info__ = ('0', '6', '0')
__version__ = '.'.join(__version_info__)


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2016
            Task: Domestic Audio Tagging
            Baseline System
            ---------------------------------------------
                Tampere University of Technology / Audio Reseach Group
                Author:  Toni Heittola ( toni.heittola@tut.fi )

            System description
                This is an baseline implementation for D-CASE 2016 challenge domestic audio tagging task.
                Features: MFCC (static+delta+acceleration)
                Classifier: GMM

        '''))

    parser.add_argument("-development", help="Use the system in the development mode", action='store_true',
                        default=False, dest='development')
    parser.add_argument("-challenge", help="Use the system in the challenge mode", action='store_true',
                        default=False, dest='challenge')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    args = parser.parse_args()

    # Load parameters from config file
    params = load_parameters('task4_audio_tagging.yaml')

    title("DCASE 2016::Domestic Audio Tagging / Baseline System")

    # Check if mode is defined
    if not (args.development or args.challenge):
        args.development = True
        args.challenge = False

    #args.development = False; args.challenge = True
    if args.development and not args.challenge:
        print "Running system in development mode"
        dataset_evaluation_mode = 'folds'
        # Get dataset container class
        dataset = eval(params['general']['development_dataset'])(data_path=params['path']['data'])        
    elif not args.development and args.challenge:
        print "Running system in challenge mode"
        dataset_evaluation_mode = 'full'
        # Get dataset container class
        dataset = eval(params['general']['challenge_dataset'])(data_path=params['path']['data'])        

    params = process_parameters(params, dataset.__class__.__name__)        

    # Fetch data over internet and setup the data
    # ==================================================
    if params['flow']['initialize']:
        dataset.fetch()

    # Extract features for all audio files in the dataset
    # ==================================================
    if params['flow']['extract_features']:
        section_header('Feature extraction')

        # Check that target path exists, create if not
        check_path(params['path']['features'])
        files = []
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            for id, item in enumerate(dataset.train(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
            for id, item in enumerate(dataset.test(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
        files = sorted(files)

        # Go through files and make sure all features are extracted
        do_feature_extraction(files=files,
                              dataset=dataset,
                              feature_path=params['path']['features'],
                              params=params['features'],
                              overwrite=params['general']['overwrite'])

        foot()

    # Prepare feature normalizers
    # ==================================================
    if params['flow']['feature_normalizer']:
        section_header('Feature normalizer')

        do_feature_normalization(dataset=dataset,
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 feature_normalizer_path=params['path']['feature_normalizers'],
                                 feature_path=params['path']['features'],
                                 overwrite=params['general']['overwrite'])

        foot()

    # System training
    # ==================================================
    if params['flow']['train_system']:
        section_header('System training')

        do_system_training(dataset=dataset,
                           dataset_evaluation_mode=dataset_evaluation_mode,
                           model_path=params['path']['models'],
                           feature_normalizer_path=params['path']['feature_normalizers'],
                           feature_path=params['path']['features'],
                           hop_length_seconds=params['features']['hop_length_seconds'],
                           classifier_params=params['classifier']['parameters'],
                           classifier_method=params['classifier']['method'],
                           overwrite=params['general']['overwrite']
                           )

        foot()

    # System evaluation in development mode
    # System testing
    # ==================================================
    if params['flow']['test_system']:
        section_header('System testing     [Development data]')

        do_system_testing(dataset=dataset,
                            dataset_evaluation_mode=dataset_evaluation_mode,
                            result_path=params['path']['results'],
                            model_path=params['path']['models'],
                            feature_path=params['path']['features'],
                            feature_params=params['features'],
                            classifier_method=params['classifier']['method'],
                            overwrite=params['general']['overwrite']
                            )
        foot()

    # System evaluation
    # ==================================================
    if args.development and not args.challenge:
        if params['flow']['evaluate_system']:
            section_header('System evaluation  [Development data]')
    
            do_system_evaluation(dataset=dataset,
                                    dataset_evaluation_mode=dataset_evaluation_mode,
                                    result_path=params['path']['results'])
    
            foot()

            
def process_parameters(params, dataset):
    params['features']['mfcc']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
    params['features']['mfcc']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])

    # Copy parameters for current classifier method
    params['classifier']['parameters'] = params['classifier_parameters'][params['classifier']['method']]

    params['features']['hash'] = get_parameter_hash(params['features'])
    params['classifier']['hash'] = get_parameter_hash(params['classifier'])

    params['path']['features'] = os.path.join(params['path']['base'], params['path']['features'], params['features']['hash'])
    params['path']['feature_normalizers'] = os.path.join(params['path']['base'], params['path']['feature_normalizers'], dataset, params['features']['hash'])
    params['path']['models'] = os.path.join(params['path']['base'], params['path']['models'], dataset, params['features']['hash'], params['classifier']['hash'])
    params['path']['results'] = os.path.join(params['path']['base'], params['path']['results'], dataset, params['features']['hash'], params['classifier']['hash'])
    return params


def get_feature_filename(audio_file, path, extension='cpickle'):
    return os.path.join(path, os.path.splitext(audio_file)[0] + '.' + extension)

def get_feature_normalizer_filename(fold, path, extension='cpickle'):
    return os.path.join(path, 'scale_fold' + str(fold)+ '.' + extension)


def get_model_filename(fold, path, extension='cpickle'):
    return os.path.join(path, 'model_fold' + str(fold) + '.' + extension)


def get_result_filename(fold, path, extension='txt'):
    return os.path.join(path, 'results_fold' + str(fold) + '.' + extension)


def do_feature_extraction(files, dataset, feature_path, params, overwrite=False):
    # Check that target path exists, create if not
    check_path(feature_path)

    for file_id, audio_filename in enumerate(files):
        # Get feature filename
        current_feature_file = get_feature_filename(audio_file=os.path.split(audio_filename)[1], path=feature_path)

        progress(title='Extracting [sequences]',
                 percentage=(float(file_id) / len(files)),
                 note=os.path.split(audio_filename)[1])

        if not os.path.isfile(current_feature_file) or overwrite:
            # Load audio
            if os.path.isfile(dataset.relative_to_absolute_path(audio_filename)):
                y, fs = load_audio(filename=dataset.relative_to_absolute_path(audio_filename), mono=True, fs=params['fs'])
            else:
                raise IOError("Audio file not found [%s]" % audio_filename)

            # Extract features
            feature_data = feature_extraction(y=y,
                                              fs=fs,
                                              include_mfcc0=params['include_mfcc0'],
                                              include_delta=params['include_delta'],
                                              include_acceleration=params['include_acceleration'],
                                              mfcc_params=params['mfcc'],
                                              delta_params=params['mfcc_delta'],
                                              acceleration_params=params['mfcc_acceleration'])
            # Save
            save_data(current_feature_file, feature_data)


def do_feature_normalization(dataset, dataset_evaluation_mode, feature_normalizer_path, feature_path, overwrite=False):
    # Check that target path exists, create if not
    check_path(feature_normalizer_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):        
        current_normalizer_file = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)
        files = []
        if not os.path.isfile(current_normalizer_file) or overwrite:
            # Initialize statistics
            for item_id, item in enumerate(dataset.train(fold)):
                if item['file'] not in files:
                    files.append(item['file'])

            file_count = len(files)
            normalizer = FeatureNormalizer()

            for file_id, audio_filename in enumerate(files):
                progress(title='Collecting data',
                         fold=fold,
                         percentage=(float(file_id) / file_count),
                         note=os.path.split(audio_filename)[1])

                # Load features
                feature_filename = get_feature_filename(audio_file=os.path.split(audio_filename)[1], path=feature_path)
                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['stat']
                else:
                    raise IOError("Features missing [%s]" % audio_filename)

                # Accumulate statistics
                normalizer.accumulate(feature_data)

            # Calculate normalization factors
            normalizer.finalize()

            # Save
            save_data(current_normalizer_file, normalizer)


def do_system_training(dataset, dataset_evaluation_mode, model_path, feature_normalizer_path, feature_path,
                       hop_length_seconds, classifier_params, classifier_method='gmm', overwrite=False):
    if classifier_method != 'gmm':
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(model_path)

    numpy.random.seed(10553)
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_model_file = get_model_filename(fold=fold, path=model_path)
        if not os.path.isfile(current_model_file) or overwrite:
            # Load normalizer
            feature_normalizer_filename = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)
            if os.path.isfile(feature_normalizer_filename):
                normalizer = load_data(feature_normalizer_filename)
            else:
                raise IOError("Feature normalizer missing [%s]" % feature_normalizer_filename)

            # Initialize model container
            model_container = {'normalizer': normalizer, 'models': {}}

            for tag_id, tag in enumerate(dataset.audio_tags):

                # Restructure training data
                positive_files = []
                negative_files = []
                for id, item in enumerate(dataset.train(fold)):
                    if tag in item['tags']:
                        positive_files.append(item)
                    else:
                        negative_files.append(item)

                # Collect positive training examples
                data_positive = None
                for id, item in enumerate(positive_files):
                    progress(title='Collecting data [positive] ',
                             fold=fold,
                             label=tag,
                             percentage=(float(id) / len(positive_files)),
                             note=os.path.split(item['file'])[1])
                    
                    # Load features
                    feature_filename = get_feature_filename(audio_file=os.path.split(item['file'])[1], path=feature_path)
                    if os.path.isfile(feature_filename):
                        feature_data = load_data(feature_filename)['feat']
                    else:
                        raise IOError("Features missing [%s]" % feature_filename)

                    # Normalize features
                    feature_data = model_container['normalizer'].normalize(feature_data)

                    # Store features per class label
                    if data_positive is None:
                        data_positive = feature_data
                    else:
                        data_positive = numpy.vstack((data_positive, feature_data))
                
                # Collect negative training examples
                data_negative = None
                for id, item in enumerate(negative_files):
                    progress(title='Collecting data [negative] ',
                             fold=fold,
                             label=tag,
                             percentage=(float(id) / len(negative_files)),
                             note=os.path.split(item['file'])[1])

                    # Load features
                    feature_filename = get_feature_filename(audio_file=os.path.split(item['file'])[1], path=feature_path)
                    if os.path.isfile(feature_filename):
                        feature_data = load_data(feature_filename)['feat']
                    else:
                        raise IOError("Features missing [%s]" % feature_filename)

                    # Normalize features
                    feature_data = model_container['normalizer'].normalize(feature_data)

                    # Store features per class label
                    if data_negative is None:
                        data_negative = feature_data
                    else:
                        data_negative = numpy.vstack((data_negative, feature_data))

                # Train models
                progress(title='Train models',fold=fold,label=tag)
                model_container['models'][tag] = {}
                model_container['models'][tag]['positive'] = mixture.GMM(**classifier_params).fit(data_positive)
                model_container['models'][tag]['negative'] = mixture.GMM(**classifier_params).fit(data_negative)

            # Save models
            save_data(current_model_file, model_container)


def do_system_testing(dataset, dataset_evaluation_mode, result_path, model_path, feature_path, feature_params, classifier_method='gmm',
                      overwrite=False):

    if classifier_method != 'gmm':
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(result_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_result_file = get_result_filename(fold=fold, path=result_path)

        if not os.path.isfile(current_result_file) or overwrite:
            results = []
            
            # Load class model container
            model_filename = get_model_filename(fold=fold, path=model_path)
            if os.path.isfile(model_filename):
                model_container = load_data(model_filename)
            else:
                raise IOError("Model file not found [%s]" % model_filename)

            file_count = len(dataset.test(fold=fold))
            for id, item in enumerate(dataset.test(fold=fold)):
                progress(title='Testing',
                         fold=fold,
                         percentage=(float(id) / file_count),
                         note=os.path.split(item['file'])[1])

                # Load features
                feature_filename = get_feature_filename(audio_file=os.path.split(item['file'])[1], path=feature_path)
                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['feat']
                else:
                    raise IOError("Features missing [%s]" % feature_filename)

                # Normalize features
                feature_data = model_container['normalizer'].normalize(feature_data)

                current_result = binary_classifier(feature_data=feature_data,
                                                 model_container=model_container)

                for label in current_result:
                    _, file_name = os.path.split(item['file'])
                    results.append((file_name, label, current_result[label] ))

            # Save testing results
            with open(current_result_file, 'wt') as f:
                writer = csv.writer(f, delimiter=',')
                for result_item in results:
                    writer.writerow(result_item)


def binary_classifier(feature_data, model_container): 
    likelihood_ratios = {}

    for label_id, label in enumerate(model_container['models']):
        positive = numpy.sum(model_container['models'][label]['positive'].score(feature_data))
        negative = numpy.sum(model_container['models'][label]['negative'].score(feature_data))

        likelihood_ratios[label] = positive - negative

    return likelihood_ratios

def do_system_evaluation(dataset, dataset_evaluation_mode, result_path):
    
    # Set warnings off, sklearn metrics will trigger warning for classes without
    # predicted samples in F1-scoring. This is just to keep printing clean.
    #warnings.simplefilter("ignore")
    
    fold_wise_class_eer = numpy.zeros((len(dataset.folds(mode=dataset_evaluation_mode)), dataset.audio_tag_count))

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        class_wise_eer       = numpy.zeros((dataset.audio_tag_count))
        results = []
        result_filename = get_result_filename(fold=fold, path=result_path)
        if os.path.isfile(result_filename):
            with open(result_filename, 'rt') as f:
                for row in csv.reader(f, delimiter=','):
                    results.append(row)
        else:
            raise IOError("Result file not found [%s]" % result_filename)

        for tag_id,tag in enumerate(dataset.audio_tags):

            y_true_binary = []
            y_true_file = []
            y_score = []
            for result in results:
                if tag == result[1]:
                    relative_path = dataset.package_list[0]['local_audio_path'].replace(dataset.local_path,'')[1:] + os.path.sep + result[0]
                    y_true_file.append(result[0])
                    if tag in dataset.file_meta(relative_path)[0]['tags']:
                        y_true_binary.append(1)
                    else:
                        y_true_binary.append(0)

                    y_score.append(float(result[2]))

            if numpy.any(y_true_binary):
                class_wise_eer[tag_id] = compute_eer(result_filename, tag, dict(zip(y_true_file, y_true_binary)))
            else:
                class_wise_eer[tag_id] = None

        fold_wise_class_eer[fold - 1 if fold > 0 else fold, :] = class_wise_eer

    print "  File-wise evaluation, over %d folds" % (dataset.fold_count)

    print "     {:20s} | {:8s}".format('Tag', 'EER')
    print "     ==============================================="
    labels = numpy.array([dataset.tagcode_to_taglabel(t) for t in dataset.audio_tags])
    for i in numpy.argsort(labels):
        print "     {:20s} | {:3.3f} ".format(labels[i],
                                                                    numpy.nanmean(fold_wise_class_eer[:,i])
                                                                    )
    print "     ==============================================="
    print "     {:20s} | {:3.3f} ".format('Mean error',
                                                      numpy.mean(numpy.nanmean(fold_wise_class_eer))
                                                      )
    # Restore warnings to default settings
    warnings.simplefilter("default")    
    
if __name__ == "__main__":
    sys.exit(main(sys.argv))
