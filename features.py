import Executesql as es
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from DataProcess import dataprocess as dp
import Executesql as ex

# data = es.origindata()
# print(len(data))

# kind_name = []
# num = []

# for kind,subkind in data.groupby(data['KIND_CODE']):
#     kind_name.append(kind)
#     num.append(len(subkind))

# print(kind_name, num)
# plt.axes(aspect=1)
# plt.pie(x=num, labels=kind_name, autopct="%.0f%%")
# plt.legend(num, loc='upper left')
# plt.title("kinds' ratio")
# plt.show()

# for kind in kind_name:
#     temp = data[data["KIND_CODE"] == kind]
#     temp['QUEUE_START_TIME'] = temp['QUEUE_START_TIME'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
#     temp['index_time'] = temp['QUEUE_START_TIME'].apply(lambda x: x[:10])
#     for index, subkind in temp.groupby([temp['MAT_CODE'], temp['index_time']]):
#         print(index, len(subkind))
#     break

# temp1 = data[data['MAT_CODE'] == '40']
# temp1['QUEUE_START_TIME'] = temp1['QUEUE_START_TIME'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
# temp1['index_time'] = temp1['QUEUE_START_TIME'].apply(lambda x: x[:10])
# temp1['ss'] = temp1['QUEUE_START_TIME'].apply(lambda x: x[11:])
# for index, subkind in temp1.groupby([temp1['MAT_CODE'], temp1['index_time']]):
#     print(index, len(subkind))
#     print(type(subkind))
#     hah = subkind.sort_values(by='ss', ascending=True)
#     plt.scatter(hah['ss'], hah['interval'])
#     plt.show()

def sub_set_classify(data):
    data = data[['TASK_ID', 'MAT_CODE', 'interval']]
    data = data[data['interval'] < 10]
    # print(data[data['interval'] == data['interval'].max()])
    data['timerange'] = data['interval'].apply(lambda x: int(math.ceil(x/(1/3))))
    # print(data['timerange'].head(100))
    list1 = []
    for name, subset in data.groupby('MAT_CODE'):
        print('%s 数量: %s' % (name, len(subset)))
        a = subset['timerange'].tolist()
        b = dp.ts(np.array(a))
        b.append(name)
        b.append(len(subset))
        list1.append(b)
        # count_set = set(a)
        # count_list = list()
        # for item in count_set:
        #     count_list.append((item, a.count(item)))
        # count_list = np.array(count_list)
        # plt.bar(count_list[:, 0], count_list[:, 1], color='b')
        # plt.title(name)
        # plt.savefig("C:/Users/10446/Desktop/queue/picture/%s.jpg" % name)
        # plt.close()
    result = dp.ts_to_dataframe(list1)
    result.to_csv('aggregation10.csv', index=False)
    print(result)


def boxplot(data):
    data = data[['TASK_ID', 'MAT_CODE', 'interval']]
    data = data[data['interval'] < 25]
    for name, subset in data.groupby('MAT_CODE'):
        print('%s 数量: %s' % (name, len(subset)))
        plt.title(name)
        plt.boxplot(x=subset['interval'], showmeans=True)
        plt.show()


if __name__ == '__main__':
    pass
    sql1 = 'select * from dispatch.t_disp_entry_queue where QUEUE_START_TIME > "2019-07-01" ' \
          'and QUEUE_START_TIME < "2019-11-01"'
    data = ex.origindata(sql1)
    sub_set_classify(data)
    # data = pd.read_csv('origin_features1.csv')
    # boxplot(data)
    # sub_set_classify(data)