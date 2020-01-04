import DBconnect as DB
import pandas as pd
from DataProcess import dataprocess as dp
from datetime import datetime



def origindata(sql):
    # 创建游标
    cursor = DB.con.cursor()
    # cursor.execute(sql)
    # result = cursor.fetchall()
    # col = cursor.description
    # columename = dp.getcolumename(col)
    # data = pd.DataFrame(result, columns=columename)
    # data.dropna(subset=['QUEUE_START_TIME', 'ENTRY_NOTICE_TIME', 'ENTRY_TIME', 'FINISH_TIME'], inplace=True)
    # data['QUEUE_START_TIME'] = data['QUEUE_START_TIME'].apply(dp.todatetime)
    # data['ENTRY_NOTICE_TIME'] = data['ENTRY_NOTICE_TIME'].apply(dp.todatetime)
    # data['ENTRY_TIME'] = data['ENTRY_TIME'].apply(dp.todatetime)
    # data['FINISH_TIME'] = data['FINISH_TIME'].apply(dp.todatetime)
    # data['interval'] = data['ENTRY_NOTICE_TIME'] - data['QUEUE_START_TIME']
    # data['interval'] = data['interval'].apply(lambda x: round(x.total_seconds() / 3600, 2))
    # '''
    # 废钢只有sub_kind_code,将废钢的sub_kind_code赋值给mat_code
    # '''
    # i = 0
    # for num in range(0, len(data)):
    #     print("%s / %s" % (i + 1, len(data)))
    #     i += 1
    #     if data.iloc[num]['KIND_CODE'] == 'FG':
    #         # print(num, data.iloc[num]['KIND_CODE'])
    #         # print(data.iloc[num]['SUB_KIND_CODE'])
    #         data.iloc[num, data.columns.get_loc('MAT_CODE')] = data.iloc[num]['SUB_KIND_CODE']
    #         # print(data.iloc[num]['MAT_CODE'])
    # columeremove = ['ID', 'TASK_ID', 'KIND_CODE', 'KIND_NAME', 'SUB_KIND_CODE',
    #                 'SUB_KIND_NAME', 'MAT_CODE', 'MAT_NAME', 'GATE_CODE', 'NET_WEIGHT', 'VENDOR', 'VENDOR_CODE',
    #                 'QUEUE_START_TIME', 'ENTRY_NOTICE_TIME', 'ENTRY_TIME', 'FINISH_TIME', 'interval']
    # data = data[columeremove]
    # print(len(data))
    # data = data[~data['MAT_CODE'].isin(['GCYK', 'GJYK', 'PSFL', '0'])]
    # data = data.fillna(0)

    '''
    等待时间必须大于0
    '''
    try:
        # 执行SQL
        cursor.execute(sql)
        result = cursor.fetchall()
        col = cursor.description
        columename = dp.getcolumename(col)
        data = pd.DataFrame(result, columns=columename)
        data.dropna(subset=['QUEUE_START_TIME', 'ENTRY_NOTICE_TIME', 'ENTRY_TIME', 'FINISH_TIME'], inplace=True)
        data['QUEUE_START_TIME'] = data['QUEUE_START_TIME'].apply(dp.todatetime)
        data['ENTRY_NOTICE_TIME'] = data['ENTRY_NOTICE_TIME'].apply(dp.todatetime)
        data['ENTRY_TIME'] = data['ENTRY_TIME'].apply(dp.todatetime)
        data['FINISH_TIME'] = data['FINISH_TIME'].apply(dp.todatetime)
        data['interval'] = data['ENTRY_NOTICE_TIME'] - data['QUEUE_START_TIME']
        data['interval'] = data['interval'].apply(lambda x: round(x.total_seconds()/3600, 2))
        '''
        废钢只有sub_kind_code,将废钢的sub_kind_code赋值给mat_code
        '''
        i = 0
        for num in range(0, len(data)):
            print("%s / %s" % (i+1, len(data)))
            i += 1
            if data.iloc[num]['KIND_CODE'] == 'FG':
                # print(num, data.iloc[num]['KIND_CODE'])
                # print(data.iloc[num]['SUB_KIND_CODE'])
                data.iloc[num, data.columns.get_loc('MAT_CODE')] = data.iloc[num]['SUB_KIND_CODE']
                # print(data.iloc[num]['MAT_CODE'])
        columeremove = ['ID', 'TASK_ID', 'KIND_CODE', 'KIND_NAME', 'SUB_KIND_CODE',
                        'SUB_KIND_NAME', 'MAT_CODE', 'MAT_NAME', 'GATE_CODE', 'NET_WEIGHT', 'VENDOR', 'VENDOR_CODE',
                        'QUEUE_START_TIME', 'ENTRY_NOTICE_TIME', 'ENTRY_TIME', 'FINISH_TIME', 'interval']
        data = data[columeremove]
        print(len(data))
        data = data[~data['MAT_CODE'].isin(['GCYK', 'GJYK', 'PSFL', '0'])]
        data = data.fillna(0)
        # data2 = data[data.isnull().values==True]
        # print(len(data2))
        # print(data2)
        # data1 = data.dropna()
        # print(len(data1))
        # print(data.head(2))
        # data.to_csv('origin_features1.csv', index=False)
        # print(len(data['MAT_CODE'].drop_duplicates()))
        # print(data.columns)
        # print(data[['interval', 'QUEUE_START_TIME', 'ENTRY_NOTICE_TIME']].head(100))
    except:
        print('fail to fetch data')
    # 关闭连接
    cursor.close()
    DB.con.close()
    return data


if __name__ == '__main__':
    sql = 'select * from dispatch.t_disp_entry_queue where QUEUE_START_TIME > "2019-08-01" ' \
          'and QUEUE_START_TIME < "2019-08-10"'
    data = origindata(sql)
    results = data.groupby('MAT_CODE')
    # for name, sub_set in results:
    #     print(name)
    subgroup = results.get_group('40')
    dp.get_mat_ts(subgroup)
