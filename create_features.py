import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')


# 构建时间序列数据
'''
n_in=t  集合t个前序的数据
n_out=t   预测t个后序的结果
dropna  是否删除有Nan的行
'''


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        # 数据整体的移动
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 获得时间序列数据的列名
def get_columns_name(data, n_in, n_out):
    n_vars = 1 if type(data) is list else data.shape[1]
    names = list()
    for i in range(n_in, 0, -1):
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    return names


# 获得同物料车辆的信息
def get_car_pre_info(data, row_info):
    # 未入场的车辆的已等待时间
    tempdata_noin = data[((row_info['QUEUE_START_TIME'] - data['QUEUE_START_TIME']).dt.total_seconds()/3600 < 24) &
                         (data['MAT_CODE'] == row_info['MAT_CODE']) &
                         (data['QUEUE_START_TIME'] < row_info['QUEUE_START_TIME'])]
    tempdata_noin['interval'] = tempdata_noin['QUEUE_START_TIME'].apply(lambda x: round((row_info['QUEUE_START_TIME'] - x).total_seconds()/3600, 2))
    # 已入场车辆的准确等待时间
    tempdata_in = data[((row_info['QUEUE_START_TIME'] - data['QUEUE_START_TIME']).dt.total_seconds() / 3600 < 24) &
                       (data['MAT_CODE'] == row_info['MAT_CODE']) &
                       (data['QUEUE_START_TIME'] < row_info['QUEUE_START_TIME']) &
                       (data['ENTRY_NOTICE_TIME'] < row_info['QUEUE_START_TIME'])]
    print(row_info['QUEUE_START_TIME'])
    tempdata = pd.concat([tempdata_in, tempdata_noin])
    tempdata = tempdata.sort_values(by=['QUEUE_START_TIME'], ascending=True)
    row = pd.DataFrame(row_info)
    row = pd.DataFrame(row.values.T, index=row.columns, columns=row.index)
    tempdata = pd.concat([tempdata, row])
    column = ['NET_WEIGHT', 'interval']
    print(len(tempdata))
    if len(tempdata) == 1:
        a = series_to_supervised(tempdata[column], 18, 1)
        a['MAT_CODE'] = None
        a['QUEUE_START_TIME'] = None
        return a
    else:
        tempdata = series_to_supervised(tempdata[column], 18, 1, False)
        a = tempdata.iloc[len(tempdata)-2]
    a = pd.DataFrame(a)
    a = pd.DataFrame(a.values.T, index=a.columns, columns=a.index)
    a['MAT_CODE'] = row_info['MAT_CODE']
    a['QUEUE_START_TIME'] = row_info['QUEUE_START_TIME']
    return a


# 读取原始数据并初步处理
def data_process():
    data = pd.read_csv('origin_features.csv')
    data = data.sort_values(by=['QUEUE_START_TIME'], ascending=True)
    data['QUEUE_START_TIME'] = pd.to_datetime(data['QUEUE_START_TIME'])
    data['ENTRY_NOTICE_TIME'] = pd.to_datetime(data['ENTRY_NOTICE_TIME'])
    data['ENTRY_TIME'] = pd.to_datetime(data['ENTRY_TIME'])
    data['FINISH_TIME'] = pd.to_datetime(data['FINISH_TIME'])
    return data


data = data_process()
name = get_car_pre_info(data, data.iloc[0])
tempdataset = pd.DataFrame(columns=name.columns)
i = 0
for index, row in data.iterrows():
    tempdataset = pd.concat([tempdataset, get_car_pre_info(data, row)])
    print("%s / %s" % (i, len(data)))
    i += 1
    if i > 2:
        break
tempdataset.dropna(how='all', axis=0)
tempdataset = tempdataset.fillna(0)
print(tempdataset)
# tempdataset.to_csv('car_features.csv', index=False)

