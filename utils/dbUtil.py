import re
import pymysql
import datetime
import traceback

# 数据库的配置项
host='127.0.0.1'
port=3306
user='root'
password='123456'
database='test'

table_name = "iat_engine_record"    # 引擎使用情况表

'''
    判断表是否存在
    :param table_name 表名
    :return True，表存在；False，表不存在
'''
def table_exists(table_name):
    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset='utf8mb4')
    cursor = conn.cursor()

    sql = "show tables;"
    cursor.execute(sql)
    tables = [cursor.fetchall()]

    cursor.close()
    conn.close()

    table_list = re.findall('(\'.*?\')',str(tables))
    table_list = [re.sub("'",'',each) for each in table_list]

    if table_name in table_list:
        return True
    else:
        return False

'''
    创建引擎使用情况表
    :return ret 0，创建成功
'''
def create_iat_engine_record_table(table_name):
    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset='utf8mb4')
    cursor = conn.cursor()

    sql = '''
        CREATE TABLE `%s`  (
            `curr_time` varchar(50) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '当前时刻，精确到s',
            `curr_engine` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '当前引擎',
            `use_status` varchar(2) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '使用状态：0，未用完;1，已用完'
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 ROW_FORMAT = Dynamic;
    ''' % (table_name)

    ret = cursor.execute(sql)

    cursor.close()
    conn.close()
    return ret

'''
    当天用完的保存的库中
'''
def saveUsed2DB(curr_time, curr_engine, use_status):
    if table_exists(table_name) is False:
        create_iat_engine_record_table(table_name)

    try:
        conn = pymysql.connect(host=host,
                               port=port,
                               user=user,
                               password=password,
                               database=database,
                               charset='utf8mb4')
        cursor = conn.cursor()

        sql = "insert into %s" % (table_name)
        sql = sql + '''
                (curr_time, curr_engine, use_status) VALUES ('%s', '%s', '%s')
              ''' % (curr_time, curr_engine, use_status)
        cursor.execute(sql)
        conn.commit()

        cursor.close()
        conn.close()
    except Exception as e:
        traceback.print_exc(e)
        conn.rollback()

        cursor.close()
        conn.close()
    return 0


'''
    获取当前天的引擎使用情况
    :return 0：引擎未使用完；1，额度已使用完；
'''
def getCurrDateStatus():
    if table_exists(table_name) is False:
        create_iat_engine_record_table(table_name)

    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset='utf8mb4')
    cursor = conn.cursor()

    now = datetime.datetime.now().strftime('%Y%m%d')
    sql = "select count(1) from %s where curr_time like '%s%%'" % (table_name, now)    # 找正常表中最大值
    # print("sql:", sql)
    cursor.execute(sql)
    results = cursor.fetchall()    # results[0], <class 'tuple'>
    print(results, type(results), len(results))
    num = results[0][0]

    cursor.close()
    conn.close()

    if num == 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    print(getCurrDateStatus())