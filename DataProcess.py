from datetime import datetime
import numpy as np
import pandas as pd


class dataprocess():
    def getcolumename(data):
        list = []
        for i in range(len(data)):
            list.append(data[i][0])
        return list

    def todatetime(str):
        return datetime.strptime(str, '%Y-%m-%d %H:%M:%S')

    def ts(data):
        return [round(data.mean(), 3), round(data.std(), 3), round(np.median(data), 3), np.quantile(data, 0.25)]

    def ts_to_dataframe(list):
        return pd.DataFrame(list, columns=['mean', 'std', 'median', 'quantitle_25', 'MAT_CODE'])

class dd():
    pass

