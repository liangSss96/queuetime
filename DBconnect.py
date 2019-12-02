import pymysql

# 设置同时数据库连接数

config = {
    'host': '47.99.118.183',
    'port': 3306,
    'user': 'root',
    'password': 'Wobugaoxing1',
    'charset': 'utf8',
    'db': 'dispatch'
}
try:
    con = pymysql.connect(**config)
except:
    print('DB connection failed')
