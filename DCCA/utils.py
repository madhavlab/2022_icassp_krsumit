import numpy as np
import matplotlib.pyplot as plt
import librosa
import sys
sys.path.append('..')

from env import ul_acc_spectrograms_path, ul_mic_spectrograms_path, show_color_text  # NOQA


class rgb():
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)


def color_text(text, rgb):
    r, g, b = rgb
    if show_color_text:
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
    else:
        return text


def split_and_save(fileName):
    split = fileName.split('_')
    split_for_Name = fileName.split('/')
    return {'file': split[-3], 'spectrogram': split[-1][:-4]}


def give_bin_number(value, numBins):
    from math import floor
    return floor(value*numBins)


def normalizeXX(eg_array):
    std = eg_array.std(axis=0)
    std = np.where(std < 0.0001, 0.0001, std)
    normed = (eg_array - eg_array.mean(axis=0)) / std
    return normed


def map_to_spec_id(idx: int, numFrames):
    from operator import mod
    from math import floor

    specNum = floor(idx / numFrames)
    frameNum = mod(idx, numFrames)
    return specNum, frameNum


def metaData_from_pred(sample_from_bin, mapping):
    # mapping is the array containing spectrogram details
    # 375 frames per spectrogram assumed
    specNo, frameNo = map_to_spec_id(sample_from_bin, 375)
    dic = mapping[specNo]
    dic['frame'] = frameNo
    return dic


def use_these_files(fileList, spec_files: np.ndarray):
    take_all = False
    if fileList == None:
        take_all = True
    final_list_spec_files = []
    for f in spec_files:
        meta = split_and_save(f)
        if take_all or meta['file'] in fileList:
            final_list_spec_files.append(f)
    return final_list_spec_files


def map_to_01(value, range_, type_of_map):
    minima, maxima = range_
    maxima += 0.001  # for not getting exactly 1.0
    if type_of_map == 'linear':
        result = 0 + (1)/(maxima-minima) * (value-minima)
        return result
    elif type_of_map == 'sigmoid':
        # default to sigmoid
        return 1/(1+np.e**(-value))
    elif type_of_map == 'concave':
        return ((value-minima)/(maxima-minima))**2
    elif type_of_map == 'convex':
        return ((value-minima)/(maxima-minima))**0.5
    else:
        raise TypeError


def metadata_to_file(metadata: tuple, channel: str):
    # metadata is index id, file no, spec no, bin no
    if channel == 'mic':
        path = ul_mic_spectrograms_path
    else:
        path = ul_acc_spectrograms_path
    path = '../' + path
    return path + f'b8p2male-b10o15female_{metadata[-3]}_SdrChannels_{metadata[-2]}.npy'


def augment_data(spectrogram, number_of_copies):
    from audiomentations import SpecCompose, SpecFrequencyMask
    augment = SpecCompose(
        [
            SpecFrequencyMask(p=0.5),
        ]
    )
    l = []
    for i in range(number_of_copies):
        l.append(augment(spectrogram))

    return l


def custom_metric(y_true, y_pred):
    from scipy.stats import pearsonr
    l = len(y_true)
    y1 = y_true[0:int(l/2)]
    y2 = y_true[int(l/2):l]
    corr, _ = pearsonr(y1, y2)
    return corr


def plotLabels_(labels, axis, idx, duration, Fs):
    axis.stem(labels)
    axis.set(title='gt: %.1f s to %.1f s and samples: %.1f to %.1f' % (
        idx*duration, (idx+1)*duration, idx*duration*Fs, (idx+1)*duration*Fs))
    return axis


def plotData(labels, channel_mic, channel_acc, idx):
    r = 3
    Fs = 24000
    duration = 4
    fig, axis = plt.subplots(nrows=r, ncols=1, figsize=(15, 8), sharex=True)
    librosa.display.specshow(
        channel_mic[idx], cmap='RdBu', y_axis='linear', sr=1, hop_length=1, x_axis='frames', ax=axis[0])
    librosa.display.specshow(
        channel_acc[idx], cmap='RdBu', y_axis='linear', sr=1, hop_length=1, x_axis='frames', ax=axis[1])
    axis[0].set(title='Mic data')
    axis[0].label_outer()
    axis[1].set(title='Acc data')
    axis[1].label_outer()

    axis[2] = plotLabels_(labels[idx], axis[2], idx, duration, Fs)
    plt.show()


def plotLabels1_(labels, axis, idx, duration, Fs):
    axis.stem(labels)
#     axis.set(title = 'prediction')
#     axis.set(title='gt: %.1f s to %.1f s and samples: %.1f to %.1f'%(idx*duration, (idx+1)*duration, idx*duration*Fs, (idx+1)*duration*Fs))
    return axis


def plotData_(labels, channel_mic, channel_acc, idx):
    r = 3
    Fs = 24000
    duration = 4
    fig, axis = plt.subplots(nrows=r, ncols=1, figsize=(12, 8), sharex=True)
    librosa.display.specshow(
        channel_mic[idx], cmap='RdBu', y_axis='linear', sr=1, hop_length=1, x_axis='frames', ax=axis[0])
    # librosa.display.specshow(channel_acc[idx], cmap = 'RdBu', y_axis='linear', sr=1, hop_length=1, x_axis='frames', ax=axis[1])
    ones = np.where(labels[idx] == 1)
    # print(ones)
    # print(ones[0].shape[0])
    # print(int(len(list(ones[0]))*0.2))

    pred = np.random.choice(list(ones[0]), int(len(list(ones[0]))*0.4))
    labels1 = np.copy(labels)
    print(labels1[idx])
    labels[idx][pred] = 0
    # print(predictions)
    print(np.sum(labels1[idx]), np.sum(labels[idx]))
    # print(len(ones[0]), len(pred))
    # print(pred)

    # axis[0].set(title='Mic spectrogram')
    axis[0].label_outer()
    axis[1].stem(labels1[idx])
    # axis[1].set(title='ground truth')
    axis[1].label_outer()
    axis[2].stem(labels[idx])
    axis[2].label_outer()

    # axis[2] = plotLabels1_(labels[idx], axis[2], idx, duration, Fs)
    fig2 = plt.figure(figsize=(2, 1))
    fig2.savefig("results.pdf", bbox_inches='tight')
    plt.show()


def plotAll(channel='laccdata', key=None, nth_spec=None):
    path = debug_path + channel
    duration = 4
    Fs = 24000
    file_ = ''
    if key != None and nth_spec != None:
        key_i = key + '_%d' % nth_spec
        file_ = key_i + '.npy'
    else:
        files = os.listdir(path=path+'/x/')
        file_ = random.choice(files)
        nth_spec = int(file_.split('_')[1])
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 8), sharex=True)
    X = np.load(path + f'/x/{file_}')
    #img = librosa.display.specshow(X,cmap = 'RdBu', y_axis='linear', sr=1, hop_length=1, x_axis='frames', ax=ax[0])
    # print(type(img))
    X = np.flip(X, 0)
    ax[0].imshow(X, aspect='auto')
    ax[0].set(title=f'X: {file_}')
    ax[0].label_outer()
    # print(X.shape)
    yd = np.load(path + f'/yd/{file_}')

    ax[1].stem(yd)
    ax[1].set(title='gt: %.1f s to %.1f s and samples: %.1f to %.1f' % (
        nth_spec*duration, (nth_spec+1)*duration, nth_spec*duration*Fs, (nth_spec+1)*duration*Fs))
    # print(yd.shape)
    plt.show()
