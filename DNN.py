import numpy as np
import pandas as pd
from data_preparation import x_data_prep as xdp
from data_preparation import y_data_prep as ydp
from data_preparation import process_received_signals_at
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, log_loss
import tensorflow as tf
import pandas as pd
import itertools
from multiprocessing import Pool



def input_fn(xs, ys):
    feature_cols = {k: tf.constant(xs[k].values)
                    for k in FEATURES}
    labels = tf.constant(ys['Concentration'].values)
    return feature_cols, labels


def DNNReg(xs_train_file, xs_test_file, ys_train_file, ys_test_file, logging=True):
    # DNNRegressor
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in FEATURES]
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[40, 60, 80, 20],
                                              model_dir="/Users/apple/Desktop/Waveform/experiment6*8/logs/DNNr")
    regressor.fit(input_fn=lambda: input_fn(pd.read_csv(xs_train_file), pd.read_csv(ys_train_file)),
                  steps=30000)
    y_r = regressor.predict(input_fn=lambda: input_fn(pd.read_csv(xs_test_file), pd.read_csv(ys_test_file)))
    predictions = list(itertools.islice(y_r, 25))
    if logging:
        print("Predictions: {}".format(str(predictions)))
    return y_r


def DNNClaf(xs_train_mt, xs_test_mt, ys_train_mt, ys_test_mt, logging=True):
    # DNNClassifier
    monitor = tf.contrib.learn.monitors.ValidationMonitor(np.array(xs_test), np.array(ys_test), every_n_steps=50)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[40, 60, 80, 20],
                                                n_classes=4,
                                                model_dir="/Users/apple/Desktop/Waveform/experiment6*8/logs/DNNc")
    classifier.fit(x=np.array(xs_train), y=np.array(ys_train), steps=30000)
    y_c = list(classifier.predict(np.array(xs_test), as_iterable=True))
    if logging:
        print('Predictions: {}'.format(str(y_c)))
    return y_c


def train_and_test():
    xs_train_file, xs_test_file, ys_train_file, ys_test_file = 'xs_train_table.csv', 'xs_test_table.csv', 'ys_train_table.csv', 'ys_test_table.csv' 
    xs_train = pd.read_csv(xs_train_file).values.tolist()
    xs_test = pd.read_csv(xs_test_file).values.tolist()
    ys_train = pd.read_csv(ys_train_file)['Concentration'].values.tolist()
    ys_test = pd.read_csv(ys_test_file)['Concentration'].values.tolist()
    
    print(DNNClaf(xs_train, xs_test, ys_train, ys_test, logging=False))
    print(DNNReg(xs_train_file, xs_test_file, ys_train_file, ys_test_file, logging=False))
    
