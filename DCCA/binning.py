"""
This file is used to bin unlabelled data by using a classifier trained on accelerometer labelled data
Binning is done by saving spectrogram details mapped to its bin number, in a csv
"""

from utils import split_and_save, give_bin_number, map_to_01, map_to_spec_id, metaData_from_pred, normalizeXX, use_these_files
import sys
import os
from math import floor
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from utils import color_text, rgb
from keras.models import Model
sys.path.append('..')
warnings.filterwarnings("once")

from env import binning_cls_path, numBins, verbose, ul_acc_spectrograms_path, begin_time, binning_type  # NOQA


if __name__ == '__main__':
    binning_cls_path = '../' + binning_cls_path

    if len(sys.argv) == 2:
        binning_cls_path = sys.argv[1]
    else:
        print(color_text(
            "You can provide the binning classifier path as an argument", rgb.RED))
    print(color_text(
        f"Accepting classfier path as {binning_cls_path}", rgb.GREEN))

    model = tf.keras.models.load_model(binning_cls_path)
    x = model.layers[-2].output  # removing output layer
    model = Model(inputs=model.input, outputs=x)

    # making the path relative to this location
    files_path = '../' + ul_acc_spectrograms_path

    unlabelled_spec_files = np.sort(os.listdir(files_path))

    # file_number_list = ['87', '89', '93'] # using only these files for binning
    file_number_list = None
    print(len(unlabelled_spec_files))
    unlabelled_spec_files = use_these_files(
        file_number_list, unlabelled_spec_files)

    print(color_text("Total number of files found: " +
          str(len(unlabelled_spec_files)), rgb.YELLOW))

    mapping = list(map(split_and_save, unlabelled_spec_files))
    print(color_text("Showing how files are mapped:", rgb.YELLOW))
    print(color_text(mapping[0:3], rgb.YELLOW))

    unlabelled_specs = []
    for f in tqdm(unlabelled_spec_files):
        spec = normalizeXX(np.load(files_path + f, allow_pickle=True))
        unlabelled_specs.append(spec)
    unlabelled_specs = np.array(unlabelled_specs)

    predictions = model.predict(unlabelled_specs)

    predictions = np.squeeze(predictions)

    spectrogram_binning_value = np.sum(predictions, axis=1)

    if verbose:
        # plotting histogram of all spectograms on the basis of how many live events they have
        plt.figure(figsize=(12, 7))
        plt.hist(spectrogram_binning_value, bins=min(
            len(spectrogram_binning_value), 300))
        plt.xlabel('Occurences of POS events in a spectogram')
        plt.ylabel('Number of frames')
        plt.title('Histogram showing logits of unlabelled frames')
        plot_path = f'../plots/Before_binning_histogram_{begin_time}.png'
        plt.savefig(plot_path)
        print(color_text("Saving plot as " + plot_path, rgb.YELLOW))

    mapping_type = binning_type
    bins = [[] for _ in range(numBins)]
    print(color_text("Empty bins: " + str(bins), rgb.GREEN))
    maximum_predicted_value = max(spectrogram_binning_value)
    minimum_predicted_value = min(spectrogram_binning_value)
    s_ = "Minimum POS detected spectogram:", minimum_predicted_value, "Maximum POS detected spectogram", maximum_predicted_value
    print(color_text(s_, rgb.YELLOW))
    for n, prediction in enumerate(spectrogram_binning_value):
        compressed_prediction = map_to_01(
            prediction, (minimum_predicted_value, maximum_predicted_value), mapping_type)
        binNum = give_bin_number(compressed_prediction, numBins)
        bins[binNum].append(n)
    final_bin_size_arr = np.array([len(x) for x in bins])
    print(color_text("Final bins size:" + str(final_bin_size_arr), rgb.GREEN))

    if verbose:
        # post binning histogram
        plt.figure(figsize=(7, 4))
        plt.bar(np.linspace(0, 1, numBins), final_bin_size_arr.T, width=0.1)
        plt.xlabel('Bins')
        plt.ylabel('Number of frames')
        plt.title(f'{mapping_type} based redistribution to bins')
        plot_path = f'../plots/After_binning_histogram_{begin_time}.png'
        plt.savefig(plot_path)
        print(color_text("Saving plot as " + plot_path, rgb.YELLOW))

    to_file_array = []
    columns = ['file', 'spectrogram', 'bin']
    for n, bin_ in enumerate(bins):
        for idx in bin_:
            dic = mapping[idx]
            to_file_array.append([dic['file'], dic['spectrogram'], n])

    df = pd.DataFrame(to_file_array, columns=columns)
    file_name = f'../data/binning/spec_binned_{mapping_type}.csv'
    df.to_csv(file_name, index=False)
    print(color_text(f"Saved as {file_name}...", rgb.GREEN))
