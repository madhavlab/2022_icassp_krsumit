import keras.backend as K
import numpy as np
import sys
sys.path.append('..')

from env import show_color_text # NOQA


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


def normalizeXX(eg_array):
    # zero mean and unit variance normalization
    normed = (eg_array - eg_array.mean(axis=0)) / eg_array.std(axis=0)
    return normed


def splitData(x, y, splitfrac):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    np.random.seed(0)
    test_idx = np.random.choice(x.shape[0], int(x.shape[0]*splitfrac))
    train_idx = np.array(list(set(np.arange(x.shape[0])) - set(test_idx)))
    for i in test_idx:
        x_test.append(x[i])
        y_test.append(y[i])
    for j in train_idx:
        x_train.append(x[j])
        y_train.append(y[j])
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


if __name__ == '__main__':
    print("Only a utility file.")
