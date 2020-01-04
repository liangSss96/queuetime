from datetime import datetime
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


class dataprocess():
    # 数据库中拉取的数据获得列名
    def getcolumename(data):
        list = []
        for i in range(len(data)):
            list.append(data[i][0])
        return list

    # string转datestamp格式
    def todatetime(str):
        return datetime.datetime.strptime(str, '%Y-%m-%d %H:%M:%S')

    # 提取物料的聚合特征
    def ts(data):
        return [round(np.mean(data), 3), round(data.std(), 3), round(np.median(data), 3), np.quantile(data, 0.75)]

    # 提取的聚合特征的集合转化
    def ts_to_dataframe(list):
        return pd.DataFrame(list, columns=['mean', 'std', 'median', 'quantitle_25', 'MAT_CODE', 'numbers'])


    # 获得品种序列的时间序列折线图
    '''
    需要排队开始时间、等待时间
    '''
    def get_mat_ts(data):
        data['QUEUE_START_TIME'] = data['QUEUE_START_TIME'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        data['range'] = data['QUEUE_START_TIME'].apply(lambda x: x[:10])
        for name, sub_set in data.groupby('range'):
            sub_set.sort_values(by=['QUEUE_START_TIME'], ascending=True, inplace=True)
            sub_set.set_index('QUEUE_START_TIME', inplace=True)
            print(name, len(sub_set))
            print(sub_set)
            plt.plot(sub_set['interval'], color='r')
            plt.show()
            # break