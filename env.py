import time
from math import floor

begin_time = floor(time.time())
verbose = True
random_state = 42
show_color_text = True

# Spectrograms
num_frames = 375

# Binning
numBins = 10
# can be convex (shifts distribution to right), concave (shifts distribution to left) or linear (leaves binning distribution same as logit distribution)
binning_type = 'linear'

# DCCA
dcca_params = {
    'in_shape1': 257,
    'in_shape2': 257,
    'out_shape1': 50,
    'out_shape2': 50,
    'num_epochs': 5,
    'learning_rate': 1e-4,
    'regularisation': 1e-3,
    'batch_size': 32,
    'num_sampled_from_each_bin': 140,
    'use_all_singular_values': True,
    'obj_r1': 1e-4,
    'obj_r2': 1e-4,
    'obj_eps': 1e-12,
    'out_dim': 10,
    'momentum': 0.9
}
dcca_test = True

# Classifier
cls_from_dcca_params = {
    'num_features': dcca_params['out_shape1'],
    'rnn_sizes': [8],
    'dropout_rate': 0.5,
    'num_classes': 1,
    'num_epochs': 10,
    'batch_size': 32,
}
cls_params = {
    'num_features': dcca_params['in_shape1'],
    'rnn_sizes': [8],
    'dropout_rate': 0.5,
    'num_classes': 1,
    'num_epochs': 50,
    'batch_size': 32,
}

# Paths
binning_cls_path = './models/CRNN_binning_cls.h5'

ul_base_path = './data/unlabelled/'
ul_acc_spectrograms_path = ul_base_path + './acc/'
ul_mic_spectrograms_path = ul_base_path + './mic/'

l_base_path = './data/labelled/'
l_mic_spectrograms_path = l_base_path + 'mic/'
l_acc_spectrograms_path = l_base_path + 'acc/'
l_labels_path = l_base_path + 'labels/'
