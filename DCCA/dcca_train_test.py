"""
This file is meant to be run at DCCA directory
"""

from utils import metadata_to_file, augment_data, custom_metric, color_text, rgb
from glob import glob
from objective import cca_loss
from math import floor
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import SGD, RMSprop
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from objective import cca_loss
from model import create_DCCA_model
import warnings
import sys
sys.path.append('..')

from env import dcca_params, verbose, random_state, dcca_test, begin_time  # NOQA

# Config
warnings.filterwarnings(action='once')
np.random.seed(random_state)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[3:], 'GPU')
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

if len(sys.argv) < 2:
    print(color_text(
        "Please add binned tuple based csv file's path as argument.\nExiting...", rgb.RED))


def check_dcca(model, X_v):
    eg = X_v.iloc[0]
    tu = (eg['file'], eg['spectrogram'], eg['bin'])
    mic_file = metadata_to_file(eg, 'mic')
    acc_file = metadata_to_file(eg, 'acc')
    X1_raw = np.load(mic_file).transpose()
    X1 = tf.expand_dims(tf.convert_to_tensor(X1_raw, dtype=float), 0)
    X2_raw = np.load(acc_file).transpose()
    X2 = tf.expand_dims(tf.convert_to_tensor(X2_raw, dtype=float), 0)
    print("X1 shape:", X1.shape)
    logits = model([X1, X2], training=False)
    logits = np.squeeze(logits)
    if verbose:
        print("Example of an output feature:")
        print(logits.shape)
        print(logits)

    corr_frames = []
    for i, logit in enumerate(logits):
        corr_frames.append(custom_metric(logit, 0))
    graph_path = f'../plots/distribution_pearsonr_corr_{begin_time}.png'
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(corr_frames)
    fig.savefig(graph_path)
    print(color_text(
        f"Saving pearsonr distribution histogram at {graph_path}", rgb.GREEN))


def test_predictions(predictions, file_, spec_):
    new = np.expand_dims(predictions, 0)
    print(new.shape)
    v = np.split(np.array(new), 2, axis=4)
    emb1 = np.squeeze(v[0])
    emb2 = np.squeeze(v[1])

    print(color_text(
        "The following values are for file: {file_}, spec: {spec_}...", rgb.YELLOW))

    print(color_text("Correlation of 0th feature with ith:", rgb.YELLOW))
    for i in range(15):
        print(np.correlate(emb1[0], emb2[i]))
    print(color_text("Correlation of ith feature with ith:", rgb.YELLOW))
    for i in range(15):
        print(np.correlate(emb1[i], emb2[i]))


def test_dcca(model, data1, data2):
    new_data = []
    pred_out = model.predict([data1, data2])
    new_data.append(pred_out)
    return new_data


if __name__ == '__main__':
    # Load binned data and sample from it uniformly
    model = create_DCCA_model(dcca_params=dcca_params)
    opt = SGD(
        learning_rate=dcca_params['learning_rate'], momentum=dcca_params['momentum'], nesterov=True)
    model.compile(loss=cca_loss(
        dcca_params['out_dim'], dcca_params['use_all_singular_values'], dcca_params), optimizer=opt)

    binned_path = sys.argv[1]
    print(color_text(f"Reading binning csv from {binned_path}", rgb.YELLOW))
    spec_metadata = pd.read_csv(binned_path)
    spec_metadata.shape
    bins = pd.unique(spec_metadata['bin'])
    if verbose:
        print(color_text("Spectrograms in each bin:", rgb.YELLOW))
        print(pd.value_counts(spec_metadata['bin']))
    binned_spec_metadata = []
    for bin_ in bins:
        # dividing into multiple dataframes
        binned_spec_metadata.append(
            spec_metadata[spec_metadata['bin'] == bin_])
    binned_spec = []
    for spec_metadata in binned_spec_metadata:
        binned = spec_metadata.sample(
            n=dcca_params['num_sampled_from_each_bin'], replace=True, random_state=random_state)
        binned = binned.reset_index()
        binned_spec.append(binned)
    binned_spec = pd.concat(binned_spec, axis=0)
    binned_spec = binned_spec.reset_index(drop=True)

    # create train_test split
    X_t, X_v = train_test_split(
        binned_spec, test_size=0.2, random_state=random_state)
    # to create a dataset
    X_train = tf.data.Dataset.from_tensor_slices(X_t)
    X_val = tf.data.Dataset.from_tensor_slices(X_v)

    X_train_batches = X_train.shuffle(
        buffer_size=1024).batch(dcca_params['batch_size'])
    X_val_batches = X_val.shuffle(
        buffer_size=1024).batch(dcca_params['batch_size'])

    # train validate loops
    total_train_losses = []
    total_eval_losses = []
    for epoch in range(dcca_params['num_epochs']):
        print(color_text("Start of epoch:" + str(epoch), rgb.BLUE))
        start_time = time.time()
        """
        Due to this reloading loop, take care not to save this file with some syntax/possible runtime error, while this file is running, because the loaded code is being updated every time this loop runs
        """
        epoch_train_loss = []
        for step, X_train_batch in enumerate(tqdm(X_train_batches)):
            X_train_1 = []
            X_train_2 = []
            for X in X_train_batch:
                filename_mic = metadata_to_file(X, 'mic')
                filename_acc = metadata_to_file(X, 'acc')
                frame_mic = augment_data(
                    np.load(filename_mic).transpose(), 1)[0]
                frame_acc = augment_data(
                    np.load(filename_acc).transpose(), 1)[0]
                X_train_1.append(frame_mic)
                X_train_2.append(frame_acc)

            X_train_1 = tf.convert_to_tensor(X_train_1)
            X_train_2 = tf.convert_to_tensor(X_train_2)
            history = model.train_on_batch(
                x=[X_train_1, X_train_2],  # multi input model
                y=tf.zeros(np.shape(X_train_1)[0]),
                reset_metrics=True
            )
            epoch_train_loss.append(history)
        print(f"Train loss in {epoch}:", np.mean(epoch_train_loss))
        total_train_losses.extend(epoch_train_loss)
        epoch_val_loss = []
        for step, X_val_batch in enumerate(tqdm(X_val_batches)):
            X_val_1 = []
            X_val_2 = []
            for X in X_val_batch:
                filename_mic = metadata_to_file(X, 'mic')
                filename_acc = metadata_to_file(X, 'acc')
                frame_mic = augment_data(
                    np.load(filename_mic).transpose(), 1)[0]
                frame_acc = augment_data(
                    np.load(filename_acc).transpose(), 1)[0]
                X_val_1.append(frame_mic)
                X_val_2.append(frame_acc)
            X_val_1 = tf.convert_to_tensor(X_val_1)
            X_val_2 = tf.convert_to_tensor(X_val_2)
            history = model.test_on_batch(
                x=[X_val_1, X_val_2],
                y=tf.zeros(np.shape(X_val_1)[0]),
                reset_metrics=True
            )
            epoch_val_loss.append(history)
        print(f"Validation loss in {epoch}:", color_text(
            str(np.mean(epoch_val_loss)), rgb.YELLOW))
        total_eval_losses.extend(epoch_val_loss)

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharey=True)
    ax[0].plot(total_train_losses)
    ax[0].set_title("Training Loss")
    ax[1].plot(total_eval_losses, 'r')
    ax[1].set_title("Validation Loss")
    plot_path = '../plots/loss_graph_' + str(begin_time)+'.png'
    fig.savefig(plot_path)
    print(color_text(f"Saved train-val loss graph in {plot_path}.", rgb.GREEN))

    model_path = f'../models/dcca_{begin_time}.h5'
    print(color_text(f"Saving model as {model_path}...", rgb.GREEN))
    model.save(model_path)

    if dcca_test:
        check_dcca(model, X_v)
        # testing on one file
        file_ = 88
        spec_ = 875
        mic_data = np.load(
            f'../data/unlabelled/mic/b8p2male-b10o15female_{file_}_SdrChannels_{spec_}.npy')
        acc_data = np.load(
            f'../data/unlabelled/acc/b8p2male-b10o15female_{file_}_SdrChannels_{spec_}.npy')
        mic_data = np.transpose(np.expand_dims(mic_data, 0), (0, 2, 1))
        acc_data = np.transpose(np.expand_dims(acc_data, 0), (0, 2, 1))

        predictions = np.array(
            test_dcca(model, mic_data, mic_data), dtype="object")
        if verbose:
            test_predictions(predictions, file_, spec_)
