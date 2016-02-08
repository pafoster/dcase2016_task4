DCASE2016 Task 4 Baseline System Implemented in Python
=========================================
[Audio Research Team / Tampere University of Technology](http://arg.cs.tut.fi/)

Author(s)
- Toni Heittola (<toni.heittola@tut.fi>, <http://www.cs.tut.fi/~heittolt/>)
- Peter Foster (<p.a.foster@qmul.ac.uk>)

# Table of Contents
1. [Introduction]
2. [Installation]
3. [Usage]
4. [System blocks]
5. [System results]
6. [System parameters]
7. [License]

1. Introduction
=================================
This document describes the Python implementation of the baseline systems for the [Detection and Classification of Acoustic Scenes and Events 2016 (DCASE2016) challenge](http://www.cs.tut.fi/sgn/arg/dcase2016/) **[task 4]** Domestic audio tagging. Overall, the challenge consists of four tasks:

1. [Acoustic scene classification](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification)
2. [Synthetic audio sound event detection](http://www.cs.tut.fi/sgn/arg/dcase2016/task-synthetic-sound-event-detection)
3. [Real life audio sound event detection](http://www.cs.tut.fi/sgn/arg/dcase2016/task-real-life-sound-event-detection)
4. [Domestic audio tagging](http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging)

The baseline systems for tasks 1, 3 and 4 share the same basic approach: MFCC-based acoustic features and a GMM-based classifier.

2. Installation
=================================

The system is developed for [Python 2.7](https://www.python.org/). Currently, the baseline systems are tested only with Linux operating systems. 

**External modules required**

[*numpy*](http://www.numpy.org/), [*scipy*](http://www.scipy.org/), [*scikit-learn*](http://scikit-learn.org/)
`pip install numpy scipy scikit-learn`

Scikit-learn 0.17 is required for the machine learning implementations.

[*PyYAML*](http://pyyaml.org/)
`pip install pyyaml`

PyYAML is required for handling the configuration files.

[*librosa*](https://github.com/bmcfee/librosa)
`pip install librosa`

Librosa is required for the feature extraction.

3. Usage
=================================

The executable for task 4 is: 

task4_audio_tagging.py*

The system has two operating modes: *Development mode* and *Challenge mode*. 

The usage parameters are shown by executing `python task4_audio_tagging.py -h`

The system parameters are defined in `task4_audio_tagging.yaml`. 

#### Development mode

In this mode the system is trained and evaluated within the development dataset. This is the default operating mode. 

To run the system in this mode:
`python task4_audio_tagging.py` 
or `python task4_audio_tagging.py -development`.

#### Challenge mode

In this mode the system is trained with all the provided development data and the challenge data is run through the developed system. Output files are generated in correct format for the challenge submission. Please note that this mode is only for use once the challenge data have been released.

To run the system in this mode:
`python task4_audio_tagging.py -challenge`.

4. System blocks
=================================

The system implements following blocks:

1. Dataset initialization 
  - Downloads the dataset from the Internet if needed
  - Extracts the dataset package if needed
  - Makes sure that the meta files are appropriately formatted

2. Feature extraction (`do_feature_extraction`)
  - Goes through all the training material and extracts the acoustic features
  - Features are stored file-by-file on the local disk (pickle file)

3. Feature normalization (`do_feature_normalization`)
  - Goes through the training material in evaluation folds, and calculates global mean and std of the data.
  - Stores the normalization factors (pickle file)

4. System training (`do_system_training`)
  - Trains the system
  - Stores the trained models and feature normalization factors together on the local disk (pickle file)

5. System testing (`do_system_testing`)
  - Goes through the testing material and does the classification / detection 
  - Stores the results (text file)

6. System evaluation (`do_system_evaluation`)
  - Does the evaluation: reads the ground truth and the output of the system and calculates evaluation metrics

5. System results
=================================

#### Task 4 - domestic audio tagging

Dataset: ** CHiME-Home-refine --  development set **
Evaluation setup: 5-fold cross-validation, 7 classes.
System main parameters: Frame size: 20 ms (50% hop size), Number of components: 8, Features: MFCC 14 static coefficients (excluding 0th coefficient)

     Tag                  | EER
     ===============================================
     adult female speech  | 0.29
     adult male speech    | 0.30  
     broadband noise      | 0.09
     child speech         | 0.20 
     other                | 0.29 
     percussive sound     | 0.25 
     video game/tv        | 0.07 
     ===============================================
     Mean error           | 0.21 

6. System parameters
=================================
The parameters are set in `task4_audio_tagging.yaml`.

**Controlling the system flow**

The blocks of the system can be controlled through the configuration file. Usually all of them can be kept on. 
    
    flow:
      initialize: true
      extract_features: true
      feature_normalizer: true
      train_system: true
      test_system: true
      evaluate_system: true

**General parameters**

The selection of used dataset.

    general:
	  development_dataset: CHiMEHome_DomesticAudioTag_DevelopmentSet
      challenge_dataset: CHiMEHome_DomesticAudioTag_ChallengeSet

      overwrite: false              # Overwrite previously stored data 

`development_dataset: CHiMEHome_DomesticAudioTag_DevelopmentSet`
: The dataset handler class used while running the system in development mode. If one wants to handle a new dataset, inherit a new class from the Dataset class (`src/dataset.py`).

`challenge_dataset: CHiMEHome_DomesticAudioTag_ChallengeSet`
: The dataset handler class used while running the system in challenge mode. If one wants to handle a new dataset, inherit a new class from the Dataset class (`src/dataset.py`).

Available dataset handler classes:
**DCASE2016**

- CHiMEHome_DomesticAudioTag_DevelopmentSet
- CHiMEHome_DomesticAudioTag_ChallengeSet

`overwrite: false`
: Switch to allow the system to overwrite existing data on disk. 

  
**System paths**

This section contains the storage paths.      
      
    path:
      data: data/

      base: system/baseline_dcase2016_task4/
      features: features/
      feature_normalizers: feature_normalizers/
      models: acoustic_models/
      results: evaluation_results/

These parameters defines the folder-structure to store acoustic features, feature normalization data, trained acoustic models and store results.

`data: data/`
: Defines the path where the dataset data is downloaded and stored. Path can be relative or absolute. 

`base: system/baseline_dcase2016_task4`
: Defines the base path where the system stores the data. Other paths are stored under this path. If specified directory does not exist it is created. Path can be relative or absolute. 

`results: evaluation_results/`
: Defines the base path where results are stored.

**Feature extraction**

This section contains the feature extraction related parameters. 

    features:
      fs: 16000
      win_length_seconds: 0.02
      hop_length_seconds: 0.01

      include_mfcc0: false          #
      include_delta: false          #
      include_acceleration: false   #

      mfcc:
        window: hamming_asymmetric  # [hann_asymmetric, hamming_asymmetric]
        n_mfcc: 14                  # Number of MFCC coefficients
        n_mels: 40                  # Number of MEL bands used
        n_fft: 1024                 # FFT length
        fmin: 0                     # Minimum frequency when constructing MEL bands
        fmax: 22050                 # Maximum frequency when constructing MEL band
        htk: false                  # Switch for HTK-styled MEL-frequency equation

      mfcc_delta:
        width: 9

      mfcc_acceleration:
        width: 9

`fs: 16000`
: Default sampling frequency. If given dataset does not fulfil this criterion the audio data is resampled.


`win_length_seconds: 0.02`
: Feature extraction frame length in seconds.
    

`hop_length_seconds: 0.01`
: Feature extraction frame hop-length in seconds.


`include_mfcc0: false`
: Switch to include zeroth coefficient of static MFCC in the feature vector


`include_delta: false`
: Switch to include delta coefficients to feature vector. Zeroth MFCC is always included in the delta coefficients. The width of delta-window is set in `mfcc_delta->width: 9` 


`include_acceleration: false`
: Switch to include acceleration (delta-delta) coefficients to feature vector. Zeroth MFCC is always included in the delta coefficients. The width of acceleration-window is set in `mfcc_acceleration->width: 9` 

`mfcc->n_mfcc: 14`
: Number of MFCC coefficients

`mfcc->fmax: 22050`
: Maximum frequency for MEL band. Usually, this is set to a half of the sampling frequency.
        
**Classification**

This section contains the frame classification related parameters. 

    classifier:
      method: gmm                   # The system supports only gmm
      parameters: !!null            # Parameters are copied from classifier_parameters based on defined method

    classifier_parameters:
      gmm:
        n_components: 8             # Number of Gaussian components
        covariance_type: diag       # Diagonal or full covariance matrix
        random_state: 0
        thresh: !!null
        tol: 0.001
        min_covar: 0.001
        n_iter: 100
        n_init: 1
        params: wmc
        init_params: wmc

`classifier_parameters->gmm->n_components: 8`
: Number of Gaussians used in the modeling.

7. License
=================================

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
