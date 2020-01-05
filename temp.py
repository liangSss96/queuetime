import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from DataProcess import dataprocess as dp



# list2 = ['a', 'b']
# temp = pd.DataFrame(list2, columns=['1'])
# print(temp)
# temp = temp.apply(lambda x: x+'s')
# list2 = np.zeros(12).reshape(2,6)
# lsit1 = [[1,2],
#          [3,4]]
# print(temp.values.shape[0])
# for index, row in temp.iterrows():
#     list[index] = [1,2,3,4,5,6]
# print((np.array(list)).reshape(-1, 1))
# temp['2'] = (np.array(list)).reshape(-1, 1)
# print(temp)
# print(temp.iloc[-1])
# print(list)
# list = pd.DataFrame(list)
# print(temp)
# print(list)
# data = pd.concat([temp, list], axis=1)
# print(data)
# from sklearn.preprocessing import OneHotEncoder

# enc = OneHotEncoder()
#
list = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
data = pd.DataFrame(list, columns=['a', 'b', 'c'])
ss = (data['a'].values).reshape(2, -1)
print(ss)
# lists = data.values
# print(dp.ts(lists[:, 2]))
# print(~(data['b'].isin([1, 0])))
# data = data[(~(data['b'].isin([1, 0])))]
# print(data)
# list = [0, 0, 3]
# data1 = pd.DataFrame(list, columns=['a'], index=[0, 1, 2])
# print(data[-1:-2:-1])
# print(data.loc[1, :])
# dd = data.values[:, :]
# print(dd.shape)
# print(type(enc.fit_transform(dd)))
# enc.transform([[0, 1, 5]]).toarray()

# data = pd.read_csv('origin_features.csv')
# print(data.values[1])

# data = pd.DataFrame(columns=['a', 'b'])
# print(data)
# data['a'] = ['s']
# print(data)

# from keras.layers import Dense
#
# layers = [Dense(2)]
# print(layers)

# def draw(data):
#     means = data.rolling(12).mean()
#     std = data.rolling(12).std()
#     plt.plot(data, color='r', label='origin')
#     plt.plot(means, color='g', label='mean')
#     plt.plot(std, color='b', label='std')
#     plt.legend()
#     plt.show()


# data = pd.read_csv('origin_features1.csv')
# columnname = ['MAT_CODE', 'QUEUE_START_TIME', 'interval']
# result = data[columnname]
# result1 = result.groupby('MAT_CODE')
# print(result1.groups)
# subgroup_40 = result1.get_group(10302000050000)
# subgroup_40.sort_values(by=['QUEUE_START_TIME'], ascending=True, inplace=True)
# # print(subgroup_40.head(10))
# process = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %')
# subgroup_40_timeseries = subgroup_40[['QUEUE_START_TIME', 'interval']]
# subgroup_40_timeseries['QUEUE_START_TIME'] = subgroup_40_timeseries['QUEUE_START_TIME'].apply\
#     (lambda x:datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
# subgroup_40_timeseries.set_index('QUEUE_START_TIME', inplace=True)
# draw(subgroup_40_timeseries.head(150))
# subgroup_40_timeseries_weight = subgroup_40_timeseries['interval'].ewm(halflife=2).mean()
# residual_error = subgroup_40_timeseries - pd.DataFrame(subgroup_40_timeseries_weight)
# plt.plot(residual_error.head(150), color='r')
# plt.plot(subgroup_40_timeseries.head(150), color='b')
# plt.show()

