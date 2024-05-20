import pymysql
import sqlite3
import pandas as pd



class DatabaseConnection:
    '''
    db_info: 

    remotedb = {
        'ip':'xx.xxx.xxx.xx',
        'port':0000,
        'user':'xxxxxx',
        'pwd':'xxxxx',
        'db':'xxxx'
    }

    localdb = {
        'path':'Data/xx.db'
    }

    return:
        pandas dataframe table
    '''

    def __init__(self, dbinfo) -> None:
        self.db_dict = dbinfo

    def GetSQL(self, sql_query, tbl_type='pddf'):

        if 'ip' in self.db_dict:
            conn = pymysql.connect(
                host=self.db_dict['ip'],
                port=self.db_dict['port'],
                user=self.db_dict['user'],
                password=self.db_dict['pwd'],
                database=self.db_dict['db']
                )
        else:
            conn = sqlite3.connect(self.db_dict['path'])

        cursor = conn.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()

        col_des = cursor.description
        col_names = [col_des[i][0] for i in range(len(col_des))]
        if tbl_type == 'pddf':
            res = pd.DataFrame(list(data), columns=col_names)
        else:
            raise NotImplementedError(
                'return table data type other than pandas dataframe not implemeted'
                )
        cursor.close()
        conn.close()

        return res
    