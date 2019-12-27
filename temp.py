import pandas as pd
import numpy as np
list = ['a', 'b']
temp = pd.DataFrame(list, columns=['1'])
temp = temp.apply(lambda x: x+'s')
list = np.zeros(12).reshape(2,6)
for index, row in temp.iterrows():
    list[index] = [1,2,3,4,5,6]
# print((np.array(list)).reshape(-1, 1))
# temp['2'] = (np.array(list)).reshape(-1, 1)
# print(temp)
# print(temp.iloc[-1])
print(list)
list = pd.DataFrame(list)
print(temp)
print(list)
data = pd.concat([temp, list], axis=1)
print(data)