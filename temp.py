import pandas as pd
import numpy as np
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
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

list = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
data = pd.DataFrame(list)
print(data.loc[1, :])
# dd = data.values[:, :]
# print(dd.shape)
# print(type(enc.fit_transform(dd)))
# enc.transform([[0, 1, 5]]).toarray()

