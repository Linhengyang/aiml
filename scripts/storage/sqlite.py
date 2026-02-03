import csv
import sqlite3
import typing as t


# 打印csv文件的前几行
def print_csv(csv_path, print_numb=5, encoding='utf-8'):
    with open(csv_path, encoding=encoding) as csv_f:
        data = csv.reader(csv_f)
        r_numb = 0
        for row in data:
            print(row)
            print("-"*150)
            r_numb = r_numb + 1
            if r_numb >= print_numb:
                break


# csv文件输出到 sqlite database文件
def csv_to_sqlite(csv_path, db_path, tbl_name, dtypes, headers:t.List[str]|bool=True, encoding='utf-8'):
    '''
    csv_path:
        .csv文件地址
    db_path:
        sqlite database .db文件地址
    tbl_name:
        表名
    dtypes:
        list of "REAL", "INTEGER", "TEXT", "BLOB"
    headers:
        default True if first line of csv file is the column names
        or:
            list of column names
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if isinstance(headers, list):
        assert len(dtypes) == len(headers), 'headers and datatypes must match in length'
    # 确认header
    elif headers:
        with open(csv_path, encoding=encoding) as csv_f:
            data = csv.reader(csv_f)
            for colnames in data:
                break
        headers = colnames
    else:
        raise ValueError('headers shall be True if first line is the header or be a list of column names')
    # create table
    cursor.execute(f'''CREATE TABLE {tbl_name} (''' + 
                    ','.join([header + ' ' + dtype + '\n' for header, dtype in zip(headers, dtypes)]) + ''')''')
    with open(csv_path, encoding=encoding) as csv_f:
        if headers: # 当第一行为列名时，跳过第一行
            next(csv_f)
        row_numb = 0 # 行计数器
        for row in csv.reader(csv_f):
            row_numb += 1
            cursor.execute(f'''INSERT INTO {tbl_name} ''' + '(' + ','.join(headers) + ')' + ' values ' +\
                            '(' + ','.join(['?']*len(row)) + ')', row)
    conn.commit()
    conn.close()

    print(f'{row_numb} rows of table {tbl_name} in {db_path} written successfully from {csv_path}')
    

def print_rows_sqlite(db_path, tbl_name, num_rows=10):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql_query = \
    '''
    SELECT *
    FROM {tbl_name}
    LIMIT {num_rows}
    '''.format(
        tbl_name=tbl_name,
        num_rows=num_rows
        )
    
    cursor.execute(sql_query)
    data = cursor.fetchall()

    col_des = cursor.description
    # col_names = [col_des[i][0] for i in range(len(col_des))]
    cursor.close()
    conn.close()

    for row in data:
        print(row)
    
    print("description:", col_des)