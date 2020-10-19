import re
import pymysql
import datetime
import traceback
from utils.dateUtil import formatTimestamp

# 数据库的配置项
host='127.0.0.1'
port=3306
user='root'
password='123456'
database='serviceguide'

# table_name = "iat_engine_record"    # 引擎使用情况表

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
def create_iat_engine_record_table():
    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset='utf8mb4')
    cursor = conn.cursor()

    sql = '''
        CREATE TABLE `iat_engine_record`  (
            `curr_time` varchar(50) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '当前时刻，精确到s',
            `curr_engine` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '当前引擎',
            `use_status` varchar(2) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '使用状态：0，未用完;1，已用完'
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 ROW_FORMAT = Dynamic;
    '''

    ret = cursor.execute(sql)

    cursor.close()
    conn.close()
    return ret

'''
    当天用完的保存的库中
'''
def saveUsed2DB(curr_time, curr_engine, use_status):
    if table_exists("iat_engine_record") is False:
        create_iat_engine_record_table()

    try:
        conn = pymysql.connect(host=host,
                               port=port,
                               user=user,
                               password=password,
                               database=database,
                               charset='utf8mb4')
        cursor = conn.cursor()

        sql = "insert into iat_engine_record"
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
    if table_exists("iat_engine_record") is False:
        create_iat_engine_record_table()

    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset='utf8mb4')
    cursor = conn.cursor()

    now = datetime.datetime.now().strftime('%Y%m%d')
    sql = "select count(1) from iat_engine_record where curr_time like '%s%%'" % (now)    # 找正常表中最大值
    # print("sql:", sql)
    cursor.execute(sql)
    results = cursor.fetchall()    # results[0], <class 'tuple'>
    # print(results, type(results), len(results))
    num = results[0][0]

    cursor.close()
    conn.close()

    if num == 0:
        return 0
    else:
        return 1


'''
    创建用户画像信息表
    :return ret 0，创建成功
'''
def create_daotai_portrait_table():
    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset='utf8mb4')
    cursor = conn.cursor()

    sql = '''
        CREATE TABLE `daotai_portrait`  (
            `source` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '来源',
            `currTime` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '当前时间',
            `daotaiID` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '导台ID', 
            `portrait_luggage` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '画像信息：行李',
            `portrait_gender` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '画像信息：性别',
            `portrait_age` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '画像信息：年龄',
            `portrait_emotion` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '画像信息：表情',
            `savefile` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '保存文件',
            `sentences` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '询问问题',
            `intention` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '意图',
            `intentionLevel` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '意图级别'
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 ROW_FORMAT = Dynamic;
    '''

    ret = cursor.execute(sql)

    cursor.close()
    conn.close()
    return ret

'''
    保存用户画像信息到库中
'''
def savePortrait2DB(portraitDict):
    if table_exists("daotai_portrait") is False:
        create_daotai_portrait_table()

    try:
        conn = pymysql.connect(host=host,
                               port=port,
                               user=user,
                               password=password,
                               database=database,
                               charset='utf8mb4')
        cursor = conn.cursor()

        portrait = portraitDict["portrait"]
        sentences = portraitDict["sentences"]
        if sentences is None or len(sentences.strip("\n")) < 1:
            return

        sql = "insert into daotai_portrait"
        sql = sql + '''
                (source, currTime, daotaiID, 
                portrait_luggage, portrait_gender, portrait_age, portrait_emotion, 
                savefile, sentences, intention, intentionLevel) 
                VALUES ('%s', '%s', '%s', 
                        '%s', '%s', '%s', '%s', 
                        '%s', '%s', '%s', '%s')
              ''' % (str(portraitDict["source"]), str(portraitDict["currTime"]), str(portraitDict["daotaiID"]),
                     str(portrait["luggage"]), str(portrait["gender"]), str(portrait["age"]), str(portrait["emotion"]),
                     str(portraitDict["savefile"]), str(sentences), str(portraitDict["intention"]), str(portraitDict["intentionLevel"]))

        # print(sql)
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
    创建语义信息表
    :return ret 0，创建成功
'''
def create_daotai_bayes_table():
    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset='utf8mb4')
    cursor = conn.cursor()

    sql = '''
        CREATE TABLE `daotai_bayes`  (
            `daotaiID` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '导台ID', 
            `sentences` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '询问问题',
            `currTime` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '当前时间',
            `intention` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '意图'
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 ROW_FORMAT = Dynamic;
    '''

    ret = cursor.execute(sql)

    cursor.close()
    conn.close()
    return ret

'''
    保存语义信息到数据库
'''
def saveYuyi2DB(yuyiDict):
    if table_exists("daotai_bayes") is False:
        create_daotai_bayes_table()

    try:
        conn = pymysql.connect(host=host,
                               port=port,
                               user=user,
                               password=password,
                               database=database,
                               charset='utf8mb4')
        cursor = conn.cursor()

        timestamp = yuyiDict["timestamp"]
        currTime = formatTimestamp(float(timestamp/1000), format="%Y-%m-%d_%H:%M:%S", ms=True)
        sentences = yuyiDict["sentences"]

        if sentences is None or len(sentences.strip("\n")) < 1:
            return

        sql = "insert into daotai_bayes"
        sql = sql + '''
                (daotaiID, sentences, currTime, intention) 
                VALUES ('%s', '%s', '%s', '%s')
              ''' % (str(yuyiDict["daotaiID"]), str(sentences), str(currTime), str(yuyiDict["intention"]))

        # print(sql)
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
    创建来人感知记录表
    :return ret 0，创建成功
'''
def create_daotai_mycoming_table():
    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset='utf8mb4')
    cursor = conn.cursor()

    sql = '''
        CREATE TABLE `daotai_mycoming`  (
            `daotaiID` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '导台ID', 
            `sentences` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '询问问题',
            `currTime` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '当前时间',
            `intention` varchar(255) CHARACTER SET utf8mb4 NULL DEFAULT NULL COMMENT '意图'
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 ROW_FORMAT = Dynamic;
    '''

    ret = cursor.execute(sql)

    cursor.close()
    conn.close()
    return ret


'''
    保存来人感知信息到库
'''
def saveMyComing2DB(commingDict):
    if table_exists("daotai_mycoming") is False:
        create_daotai_mycoming_table()

    try:
        conn = pymysql.connect(host=host,
                               port=port,
                               user=user,
                               password=password,
                               database=database,
                               charset='utf8mb4')
        cursor = conn.cursor()

        timestamp = commingDict["timestamp"]
        currTime = formatTimestamp(float(int(timestamp)/1000), format="%Y-%m-%d_%H:%M:%S", ms=True)
        sentences = commingDict["sentences"]

        if sentences is None or len(sentences.strip("\n")) < 1:
            return

        sql = "insert into daotai_mycoming"
        sql = sql + '''
                (daotaiID, sentences, currTime, intention) 
                VALUES ('%s', '%s', '%s', '%s')
              ''' % (str(commingDict["daotaiID"]), str(sentences), str(currTime), str(commingDict["intention"]))

        # print(sql)
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

if __name__ == '__main__':
    # print(getCurrDateStatus())
    # portraitDict = {'source': 'yuyi', 'currTime': '2020-10-19_15:53:33.367', 'daotaiID': 'center01', 'portrait': {'luggage': [], 'gender': 'M', 'age': 22, 'emotion': 'happy'}, 'savefile': 'D:/daotai/portrait_imgs/center01_20201019155333367000.jpg', 'sentences': '我想去卫生间呀', 'intention': '找卫生间', 'intentionLevel': '1'}
    # savePortrait2DB(portraitDict)
    #
    # yuyiDict = {'daotaiID': 'center01', 'sentences': '都有这个210', 'timestamp': 1603090697807, 'intention': '听不懂'}
    # saveYuyi2DB(yuyiDict)

    commingDict = {'daotaiID': 'center01', 'sentences': 'M,21,246,229,390,373,1000,480,640', 'timestamp': '1603098833034', 'intention': 'mycoming'}
    saveMyComing2DB(commingDict)
