import DBconnect as DB
import pandas as pd
from DataProcess import dataprocess as dp


def origindata():
    sql = 'select * from dispatch.t_disp_entry_queue where QUEUE_START_TIME > "2019-10-01" limit 1000'
    # 创建游标
    cursor = DB.con.cursor()
    try:
        # 执行SQL
        cursor.execute(sql)
        result = cursor.fetchall()
        col = cursor.description
        columename = dp.getcolumename(col)
        data = pd.DataFrame(result, columns=columename)
        data['QUEUE_START_TIME'] = data['QUEUE_START_TIME'].apply(dp.todatetime)
        data['ENTRY_NOTICE_TIME'] = data['ENTRY_NOTICE_TIME'].apply(dp.todatetime)
        data['ENTRY_TIME'] = data['ENTRY_TIME'].apply(dp.todatetime)
        data['FINISH_TIME'] = data['FINISH_TIME'].apply(dp.todatetime)
    except:
        print('fail to fetch data')
    # 关闭连接
    cursor.close()
    DB.con.close()
    return data




