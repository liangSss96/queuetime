from datetime import datetime


class dataprocess():
    def getcolumename(data):
        list = []
        for i in range(len(data)):
            list.append(data[i][0])
        return list

    def todatetime(str):
        return datetime.strptime(str, '%Y-%m-%d %H:%M:%S')

class dd():
    pass

