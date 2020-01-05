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

data = pd.read_csv('origin_features_3.csv')
data = data[data['MAT_CODE'] == 10301000010000]
data_predict = data[(data['QUEUE_START_TIME'] > '2019-08-20') &
                    (data['QUEUE_START_TIME'] < '2019-08-28')]

model = joblib.load('lasso.pkl')

# 对排在前面的每一个进行预测，直到自己
def predict_time(data, row_info):
    # 未入场的车辆的已等待时间
    tempdata_noin = data[(data['MAT_CODE'] == row_info['MAT_CODE']) &
                         (data['QUEUE_START_TIME'] < row_info['QUEUE_START_TIME']) &
                         (data['ENTRY_NOTICE_TIME'] > row_info['QUEUE_START_TIME']) &
                         (data['GATE_CODE']) == row_info['GATE_CODE']]
    # 已入场车辆的准确等待时间
    tempdata_in = data[(data['MAT_CODE'] == row_info['MAT_CODE']) &
                       (data['QUEUE_START_TIME'] < row_info['QUEUE_START_TIME']) &
                       (data['ENTRY_NOTICE_TIME'] < row_info['QUEUE_START_TIME'])]
    # print(tempdata_noin)
    row = pd.DataFrame(row_info)
    row = pd.DataFrame(row.values.T, index=row.columns, columns=row.index)
    tempdata_noin = pd.concat([tempdata_noin, row])
    tempdata_noin = tempdata_noin.sort_values(by=['QUEUE_START_TIME'], ascending=True)
    tempdata_in = tempdata_in.sort_values(by=['QUEUE_START_TIME'], ascending=True)
    # print(tempdata_noin)
    tempdata_in = tempdata_in.iloc[-8:]
    # print(tempdata_in)
    for num in range(0, len(tempdata_noin)):
        # print(tempdata_in['interval'])
        pre = (np.array(tempdata_in['interval'])).reshape(1, -1)
        # print(pre)
        # print('*********')
        # print(model.predict(pre))
        tempdata_noin.iloc[num, tempdata_in.columns.get_loc('interval')] = round(model.predict(pre)[0], 3)
        # print('----------')
        row1 = pd.DataFrame(tempdata_noin.iloc[num])
        row1 = pd.DataFrame(row1.values.T, index=row1.columns, columns=row1.index)
        # print(row1)
        # print('*********')
        tempdata_in = pd.concat([tempdata_in, row1])
        # print(tempdata_in)
        # print('/////////')
        tempdata_in = tempdata_in.iloc[-8:]
        # print(tempdata_in)
        # print((tempdata_in.iloc[-1])['interval'])
        # print('........')
        # print(round(row_info['interval'], 3))
        return [(tempdata_in.iloc[-1])['interval'], round(row_info['interval'], 3)]

if __name__ == '__main__':
    list = []
    n = 1
    j = len(data_predict)
    for index, row in data_predict.iterrows():
        print('%s / %s' % (n, j))
        list.append(predict_time(data, row))
        n += 1
    ab = pd.DataFrame(list, columns=['a', 'b'])
    cha = [abs(x) for x in (ab['a'] - ab['b'])]
    i = 0
    print(np.mean(cha))
    for num in cha:
        if num < 0.5:
            i += 1
    print(i / j)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ab['b'], label="Real")
    ax.legend(loc='upper left')
    plt.plot(ab['a'], label="Prediction")
    ax.legend(loc='upper left')
    # plt.plot(predicted_half, label="Prediction_half")
    # ax.legend(loc='upper left')
    # plt.plot(predicted_one, label="Prediction_one")
    # plt.legend(loc='upper left')
    plt.show()
