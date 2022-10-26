import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from audiomentations import SpecCompose, SpecChannelShuffle, SpecFrequencyMask
from utils import plotAll, color_text, rgb
from librosa import display
import sys
from tqdm import tqdm
sys.path.append('..')

from env import verbose, l_base_path, l_mic_spectrograms_path, l_acc_spectrograms_path, l_labels_path  # NOQA


save_path = '../data/embeddings/'

#! remove these paths below:
# l_mic_spectrograms_path = '/hdd_storage/users/sumitk/dataset/labelled_spectrograms/lmicdata/x/'
# l_acc_spectrograms_path = '/hdd_storage/users/sumitk/dataset/labelled_spectrograms/laccdata/x/'
# l_labels_path = '/hdd_storage/users/sumitk/dataset/labelled_spectrograms/lmicdata/yd/'
# l_base_path = '/hdd_storage/users/sumitk/dataset/labelled_spectrograms/'

save_path = '../data/embeddings/'

label_list = os.listdir(l_labels_path)


def loadData(mic_spec_path, acc_spec_path, label_path):
    label_list = os.listdir(label_path)
    non_zero_spec_path = []
    c = 0
    X_mic = []
    X_acc = []
    Y = []
    for file in tqdm(range(len(label_list))):
        # print(label_path+label_list[file])
        y = np.load(label_path+label_list[file])
        if (sum(y) != 0):
            # c+=1
            # print(label_path+file,spec_path+file)
            Y.append(y)
            X_mic.append(np.load(mic_spec_path+label_list[file]))
            X_acc.append(np.load(acc_spec_path+label_list[file]))
            # X.append(pad_along_axis(np.load(spec_path+label_list[file]), 46, axis = 1))
            non_zero_spec_path.append(mic_spec_path+label_list[file])
            # X.append(np.load(spec_path+label_list[file]))
    # print(c)
    return np.array(X_mic), np.array(X_acc), np.array(Y), non_zero_spec_path


def augment_data(idx, number_of_copies):
    augment = SpecCompose(
        [
            # SpecChannelShuffle(p=0.5),
            SpecFrequencyMask(p=0.5),
        ]
    )
    print('Total number of copies: ', len(idx)*number_of_copies)
    # Example spectrogram with 1025 frequency bins, 256 time steps and 2 audio channels
    mic_spects_ = []
    acc_spects_ = []
    for i in range(len(idx)):
        for j in range(number_of_copies):
            mic_spectrogram = mic_spec[idx[i]]
            # Augment/transform/perturb the spectrogram
            augmented_mic_spectrogram = augment(mic_spectrogram)
            mic_spects_.append(augmented_mic_spectrogram)
            acc_spectrogram = acc_spec[idx[i]]
            # Augment/transform/perturb the spectrogram
            augmented_acc_spectrogram = augment(acc_spectrogram)
            acc_spects_.append(augmented_acc_spectrogram)

    return mic_spects_, acc_spects_


def augment_labels(idx, number_of_copies):
    # add augmented labels to previous labels
    aug_lables = []
    for i in range(len(idx)):
        # aug_lables_ = []
        a = np.tile(label[idx[i]], number_of_copies).reshape(
            number_of_copies, mic_spectrograms.shape[2])
        # print(a.shape)
        aug_lables.append(a)

    return np.array(aug_lables)


if __name__ == '__main__':
    # try:
    #     if verbose:
    #         plotAll('lmicdata')
    # except:
    #     pass
    print(color_text(
        f"Loading mic, acc, and label data from {l_base_path}...", rgb.YELLOW))
    mic_spec, acc_spec, label, nonzero_spec_path = loadData(
        l_mic_spectrograms_path, l_acc_spectrograms_path, l_labels_path)
    print(mic_spec.shape, acc_spec.shape, label.shape)

    large_pos_specs = []
    for i in range(len(label)):
        if np.sum(label[i]) > 200:
            large_pos_specs.append(i)
    if verbose:
        print(color_text("Augmenting data now...", rgb.YELLOW))
    mic_spec_copies, acc_spec_copies = augment_data(large_pos_specs, 25)
    mic_spectrograms = np.concatenate((mic_spec, np.array(mic_spec_copies)))
    acc_spectrograms = np.concatenate((acc_spec, np.array(acc_spec_copies)))
    del mic_spec
    del acc_spec
    augmented_labels = augment_labels(large_pos_specs, 25)
    x = augmented_labels.reshape(
        augmented_labels.shape[0]*augmented_labels.shape[1], augmented_labels.shape[2])
    labels_aug = np.concatenate((label, x))
    print(f"Final Mic shape : {mic_spectrograms.shape}")
    print(f"Final Acc shape : {acc_spectrograms.shape}")
    print(f"Final Labels shape : {labels_aug.shape}")

    np.save(save_path + 'merged_mic_spectrograms_', mic_spectrograms)
    np.save(save_path + 'merged_acc_spectrograms_', acc_spectrograms)
    np.save(save_path + 'merged_labels_', labels_aug)
    print(color_text(
        f"Saved merged spectrograms and labels to {save_path}", rgb.GREEN))
