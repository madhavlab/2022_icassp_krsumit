# Run as:
# python create_embeddings.py ../models/dcca_with_binning.h5 ../data/embeddings/old_merged_acc_spectrograms_.npy ../data/embeddings/old_merged_mic_spectrograms_.npy ../data/embeddings/old_merged_labels_.npy

from keras.optimizers import SGD
from tensorflow import keras
import tensorflow as tf
import numpy as np
from objective import cca_loss
import warnings
import sys
from utils import color_text, rgb
sys.path.append('..')

# Config
warnings.filterwarnings(action='once')
tf.config.run_functions_eagerly(True)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[2:], 'GPU')

from env import dcca_params, begin_time  # NOQA

if len(sys.argv) < 4:
    print(color_text(
        "Provide path to [0] model, .npy of merged spectrograms of [1] mic, [2] acc respectively.\nExiting...", rgb.RED))
    exit()

model_path = sys.argv[1]
mic_merged_path = sys.argv[2]
acc_merged_path = sys.argv[3]
# labels_merged_path = sys.argv[4]


def test_model(model, data1: np.ndarray, data2: np.ndarray):
    # producing the new features
    new_data = []
    pred_out = model.predict([data1, data2])
    new_data.append(pred_out)
    return new_data


if __name__ == '__main__':
    model = keras.models.load_model(
        model_path, compile=False)
    opt = SGD(
        learning_rate=dcca_params['learning_rate'], momentum=0.9, nesterov=True)
    model.compile(loss=cca_loss(
        dcca_params['out_dim'], dcca_params['use_all_singular_values'], dcca_params), optimizer=opt)
    mic_data = np.transpose(np.load(mic_merged_path), (0, 2, 1))
    acc_data = np.transpose(np.load(acc_merged_path), (0, 2, 1))
    # labels = np.load(labels_merged_path)
    assert mic_data.shape[0] == acc_data.shape[0]
    # assert mic_data.shape[0] == labels.shape[0]
    new = np.array(test_model(model, mic_data, mic_data), dtype="object")
    embeddings_path = f'../data/embeddings/dcca_created_embeddings_{begin_time}'
    np.save(embeddings_path, new)
    print(color_text(
        f"Saved embeddings created via DCCA as {embeddings_path}", rgb.GREEN))
