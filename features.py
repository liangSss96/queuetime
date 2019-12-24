import Executesql as es
import matplotlib.pyplot as plt


data = es.origindata()
print(len(data))

kind_name = []
num = []


for kind,subkind in data.groupby(data['KIND_CODE']):
    kind_name.append(kind)
    num.append(len(subkind))

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

temp1 = data[data['MAT_CODE'] == '40']
temp1['QUEUE_START_TIME'] = temp1['QUEUE_START_TIME'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
temp1['index_time'] = temp1['QUEUE_START_TIME'].apply(lambda x: x[:10])
temp1['ss'] = temp1['QUEUE_START_TIME'].apply(lambda x: x[11:])
for index, subkind in temp1.groupby([temp1['MAT_CODE'], temp1['index_time']]):
    print(index, len(subkind))
    print(type(subkind))
    hah = subkind.sort_values(by='ss', ascending=True)
    plt.scatter(hah['ss'], hah['interval'])
    plt.show()