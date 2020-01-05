import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers.core import Dropout, Dense, Activation
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')



dataset = pd.read_csv('car_features_true_time.csv')


def bulid_features_RNN(data):
    # print(type(data), '\n', data.columns)
    print(type(data['MAT_CODE'][0]))
    data_40 = data[data['MAT_CODE'] == 40].values
    # data_40 = data.values
    print(data_40.shape)
    data_40 = data_40[:, 0:-2]
    row = int(round(data_40.shape[0] * 0.7))
    train = data_40[100:row, :]
    X_train = train[:, 10:-1]
    Y_train = train[:, -1]
    X_test = data_40[row:, 10:-1]
    Y_test = data_40[row:, -1]
    print(X_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 8))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 8))
    return [X_train, Y_train, X_test, Y_test]


def bulid_features_NN(data):
    # print(type(data), '\n', data.columns)
    print(type(data['MAT_CODE'][0]))
    data_40 = data[data['MAT_CODE'] == 40].values
    # data_40 = data.values
    print(data_40.shape)
    data_40 = data_40[:, 0:-2]
    row = int(round(data_40.shape[0] * 0.8))
    train = data_40[100:row, :]
    X_train = train[:, 10:-1]
    Y_train = train[:, -1]
    X_test = data_40[row:, 10:-1]
    Y_test = data_40[row:, -1]
    print(X_train.shape)
    # X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], 1, 1))
    return [X_train, Y_train, X_test, Y_test]



def bulid_model_RNN():
    model = Sequential()
    model.add(GRU(10, input_shape=(None, 8),
                   return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(20,
                   return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mape'])
    print("compilation Time : ", time.time() - start)
    return model


def bulid_NN():
    model = Sequential()
    # model.add(Dense(10, input_dim=8, activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(1, input_dim=8))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mape'])
    print("compilation Time : ", time.time() - start)
    return model

def run_model_RNN(model=None, data=None):
    epochs = 2
    ratio = 1
    sequence_length = 20
    X_train, y_train, X_test, y_test = bulid_features_RNN(dataset)
    model = bulid_model_RNN()
    model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_split=0.05)
    joblib.dump(model, 'RNN.pkl')
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))
    predicted_half = [x + 0.5 for x in predicted]
    predicted_one = [x + 1 for x in predicted]
    scores = model.evaluate(X_test, y_test, batch_size=128)
    print("\nevaluate result: \nmse={:.6f}\nmae={:.6f}\nmape={:.6f}".format(scores[0], scores[1], scores[2]))
    cha = [abs(x) for x in (predicted - y_test)]
    i=0
    for num in cha:
        if num < 0.5:
            i += 1
    print(i / len(y_test))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test, label="Real")
    ax.legend(loc='upper left')
    plt.plot(predicted, label="Prediction")
    ax.legend(loc='upper left')
    plt.plot(predicted_half, label="Prediction_half")
    ax.legend(loc='upper left')
    plt.plot(predicted_one, label="Prediction_one")
    plt.legend(loc='upper left')
    y = y_test.tolist()
    y.insert(0, 0)
    plt.plot(y, label="Predictiondddd")
    plt.legend(loc='upper left')
    plt.show()


def run_NN(model=None, data=None):
    epochs = 2
    X_train, y_train, X_test, y_test = bulid_features_NN(dataset)
    model = bulid_NN()
    model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_split=0.05)
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))
    predicted_half = [x + 0.5 for x in predicted]
    predicted_one = [x + 1 for x in predicted]
    scores = model.evaluate(X_test, y_test, batch_size=128)
    print("\nevaluate result: \nmse={:.6f}\nmae={:.6f}\nmape={:.6f}".format(scores[0], scores[1], scores[2]))
    cha = [abs(x) for x in (predicted - y_test)]
    i = 0
    for num in cha:
        if num < 0.5:
            i += 1
    print(i / len(y_test))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test, label="Real")
    ax.legend(loc='upper left')
    plt.plot(predicted, label="Prediction")
    ax.legend(loc='upper left')
    # plt.plot(predicted_half, label="Prediction_half")
    # ax.legend(loc='upper left')
    # plt.plot(predicted_one, label="Prediction_one")
    # plt.legend(loc='upper left')
    plt.show()


def linear_lasso():
    model = LassoCV(
        alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10],
        # alphas=np.arange(0.001, 0.02, 0.001),
        normalize=False,
        precompute='auto',
        cv=2,
        max_iter=500
    )
    X_train, y_train, X_test, y_test = bulid_features_NN(dataset)
    model.fit(X_train, y_train)
    joblib.dump(model, 'lasso.pkl')
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))
    predicted_half = [x + 0.5 for x in predicted]
    predicted_one = [x + 1 for x in predicted]
    # scores = model.evaluate(X_test, y_test, batch_size=128)
    # print("\nevaluate result: \nmse={:.6f}\nmae={:.6f}\nmape={:.6f}".format(scores[0], scores[1], scores[2]))
    cha = [abs(x) for x in (predicted - y_test)]
    i = 0
    for num in cha:
        if num < 0.5:
            i += 1
    print(i / len(y_test))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test, label="Real")
    ax.legend(loc='upper left')
    plt.plot(predicted, label="Prediction")
    ax.legend(loc='upper left')
    # plt.plot(predicted_half, label="Prediction_half")
    # ax.legend(loc='upper left')
    # plt.plot(predicted_one, label="Prediction_one")
    # plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    # run_model_RNN()
    # print(dataset['MAT_CODE'])
    run_NN()
    linear_lasso()
    # model = joblib.load('lasso.pkl')
    # print(model.predict((np.array([1,2,3,4,5,6]).reshape(1, -1))))
