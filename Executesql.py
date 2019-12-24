import DBconnect as DB
import pandas as pd
from DataProcess import dataprocess as dp
from datetime import datetime


def origindata():
    i=0
    sql = 'select * from dispatch.t_disp_entry_queue where QUEUE_START_TIME > "2019-08-01" ' \
          'and QUEUE_START_TIME < "2019-09-01"'
    # 创建游标
    cursor = DB.con.cursor()
    # 执行SQL
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
    # data['interval'] = (data['ENTRY_NOTICE_TIME'] - data['QUEUE_START_TIME'])
    # data['interval'] = data['interval'].apply(lambda x: x.total_seconds()/3600)
    # # data['MAT_CODE'] = data.apply(lambda row: dp.change(row['KIND_CODE'], row['MAT_CODE'], row['SUB_KIND_CODE'], axis= 1))
    #
    # for num in range(0, len(data)):
    #     if data.iloc[num]['KIND_CODE'] == 'FG':
    #         # print(num, data.iloc[num]['KIND_CODE'])
    #         # print(data.iloc[num]['SUB_KIND_CODE'])
    #         data.iloc[num, data.columns.get_loc('MAT_CODE')] = data.iloc[num]['SUB_KIND_CODE']
    #         # print(data.iloc[num]['MAT_CODE'])
    #
    # print(data['interval'])
    # # for index, row in data.iterrows():   //只读
    # #     if row['KIND_CODE'] == 'FG':
    # #         row['MAT_CODE'] = row['SUB_KIND_CODE']
    # # print(data['MAT_CODE'].head(100))
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
        data['interval'] = data['interval'].apply(lambda x: x.total_seconds()/3600)
        '''
        废钢只有sub_kind_code,将废钢的sub_kind_code赋值给mat_code
        '''
        for num in range(0, len(data)):
            if data.iloc[num]['KIND_CODE'] == 'FG':
                # print(num, data.iloc[num]['KIND_CODE'])
                # print(data.iloc[num]['SUB_KIND_CODE'])
                data.iloc[num, data.columns.get_loc('MAT_CODE')] = data.iloc[num]['SUB_KIND_CODE']
                # print(data.iloc[num]['MAT_CODE'])
        columeremove = ['ID', 'TASK_ID', 'DEAL_ID', 'KIND_CODE', 'KIND_NAME', 'SUB_KIND_CODE',
       'SUB_KIND_NAME', 'MAT_CODE', 'MAT_NAME', 'TRUCK_KIND',
       'GATE_CODE', 'NET_WEIGHT', 'VENDOR','VENDOR_CODE',
       'QUEUE_START_TIME', 'ENTRY_NOTICE_TIME', 'ENTRY_TIME', 'FINISH_TIME', 'interval']
        data = data[columeremove]
        print(len(data['MAT_CODE'].drop_duplicates()))
        print(data.columns)
        # print(data[['interval', 'QUEUE_START_TIME', 'ENTRY_NOTICE_TIME']].head(100))
    except:
        print('fail to fetch data')
    # 关闭连接
    cursor.close()
    DB.con.close()
    return data


if __name__ == '__main__':
    origindata()

