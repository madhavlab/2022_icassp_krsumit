from utils import color_text, normalizeXX, splitData, rgb
import keras_metrics
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_metrics
import warnings
import keras.backend as K
from model import create_model
import sys
sys.path.append('..')

from env import cls_from_dcca_params, cls_params, random_state, begin_time, num_frames  # NOQA

# Config
warnings.filterwarnings(action='once')
np.random.seed(seed=random_state)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[2:], 'GPU')
# tf.config.run_functions_eagerly(True)


if len(sys.argv) < 3:
    print(color_text("Please include labelled feature file and label file paths as arguments respectively.\nExiting...", rgb.RED))
    exit()

data = np.load(sys.argv[1], allow_pickle=True)

is_dcca_based = input(color_text(
    "Are these DCCA embeddings? (Y/N): ", rgb.RED))
if is_dcca_based.lower() == 'y':
    is_dcca_based = True
else:
    is_dcca_based = False

feature_dict = cls_from_dcca_params if is_dcca_based else cls_params

epochs = feature_dict['num_epochs']
num_classes = feature_dict['num_classes']
batch_size = feature_dict['batch_size']
rnn_sizes = feature_dict['rnn_sizes']
dropout_rate = feature_dict['dropout_rate']
num_features = feature_dict['num_features']
data = np.squeeze(data)

# below would either be 2*50 or 257
reshape_size = 2*num_features if is_dcca_based else num_features
data = np.reshape(
    data, (-1, num_frames, reshape_size))

if is_dcca_based:
    v = np.split(np.array(data), 2, axis=2)
    X = v[0]  # we'll only use the first set of embeddings
    # X = v[1] #! you can try out with the second set
else:
    X = data

X = np.transpose(X, (0, 2, 1))
print(color_text(X.shape, rgb.YELLOW))

Y = np.load(sys.argv[2], allow_pickle=True)

X_train, X_test, t_train, t_test = train_test_split(
    X, Y, test_size=0.1, random_state=random_state)
X_Train, X_val, t_Train, t_val = train_test_split(
    X_train, t_train, test_size=1/9, random_state=random_state)


print("Train X Shape:", X_Train.shape,  # type: ignore
      "Train label Shape:", t_Train.shape,  # type: ignore
      "\nVal X Shape:", X_val.shape,  # type: ignore
      "Val label Shape:", t_val.shape,  # type: ignore
      "\nTest X Shape:", X_test.shape,  # type: ignore
      "Test label Shape:", t_test.shape)  # type: ignore


# calculating class weights
num_ones = (np.count_nonzero(np.array(Y)))
num_zeros = Y.shape[0]*Y.shape[1] - num_ones
one_weight = np.abs((num_ones/(num_ones + num_zeros)))
zero_weight = np.abs((num_zeros/(num_ones + num_zeros)))
print(color_text(
    f"Duty cycle of these spectograms together: {one_weight}", rgb.YELLOW))


# calculating sample weights
weight = np.where(t_Train == 0, zero_weight, t_Train)
weights = np.where(weight == 1, one_weight, weight)
weights = np.squeeze(weights)


t_Train = t_Train[..., None]  # type: ignore
t_test = t_test[..., None]  # type: ignore


model = create_model(feature_dict['num_features'],
                     rnn_sizes,
                     dropout_rate,
                     num_classes)

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=[
              keras.metrics.Precision(), keras.metrics.Recall()], sample_weight_mode="temporal")

print(color_text("Training phase starting...", rgb.GREEN))
train = model.fit(tf.convert_to_tensor(X_Train, dtype=tf.float32),
                  tf.convert_to_tensor(t_Train, dtype=tf.float32),
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(tf.convert_to_tensor(
                      X_val, dtype=tf.float32), tf.convert_to_tensor(t_val, dtype=tf.float32)),
                  sample_weight=weights)

print(color_text("Testing phase starting...", rgb.GREEN))
test = model.evaluate(tf.convert_to_tensor(
    X_test, dtype=tf.float32), tf.convert_to_tensor(t_test, dtype=tf.float32))
loss = test[0]
p = test[1]
r = test[2]
f1 = 2*p*r/(p+r)
test.append(f1)
print(color_text(
    f"RESULTS:\nLoss: {loss}, precision: {p}, recall: {r}, f1 score: {f1}", rgb.GREEN))

model_path = f'../models/CRNN_cls_{"dcca_emb" if is_dcca_based else "spec"}_{f1}_{begin_time}.h5'
model.save(model_path)
print(color_text(f"Saved model at {model_path}", rgb.GREEN))
