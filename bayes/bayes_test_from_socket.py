# -*- coding:utf-8 -*-

import os
import sys
import time
import json
import random
import socket
import configparser
import traceback
import threading
import itertools
import pika
from sklearn.externals import joblib

base_path = "D:/workspace/workspace_python/daotai-semantics"
sys.path.append(base_path)
from bayes.bayes_train import get_words, bernousNB_save_path, isChat
from utils.commonutil import getFormatTime, resolving_recv
from utils.dbUtil import saveYuyi2DB, saveUsed2DB, savePortrait2DB
from config import daotaiID

sys.path.append("..")
from Logger import *

'''
    从文件读取模型并进行分类，打开socket，接收消息
'''

AnswerDict = []
intentionList = []
ask_sentenses_length = 5    # 当未包含关键字，且问话>5个字时，认为需要转接人工了

# 日志
# semantics_logfile = 'D:/data/daotai_semantics.log'
# semantics_log = Logger(semantics_logfile, level='info')
#
# bayes_mq_logfile = 'D:/data/daotai_bayes_mq.log'
# bayes_mq_log = Logger(bayes_mq_logfile, level='info')

def loadAnswers():
    with open("../kdata/intention_answer.txt", encoding="utf-8", errors="ignore") as fo:
        for line in fo.readlines():
            arr = line.strip().split("\t")
            AnswerDict[arr[0]] = arr[2]
            intentionList.append(arr[0])
    print("load answers finished")

def getAnswer(intention):
    result_list = AnswerDict[intention].split("|")
    return result_list[random.randint(0, len(result_list) - 1)]

'''
    获取最新的模型
'''
def get_newest_model(model_path):
    model_full_path = os.path.join(os.path.dirname(__file__), model_path)
    model_full_path = model_full_path.replace('\\', '').replace('.', '')

    if os.path.exists(model_full_path):
        # 按文件最后修改时间排序，reverse=True表示降序排序
        filelist = sorted(os.listdir(model_full_path), key=lambda x: os.path.getctime(os.path.join(model_full_path, x)), reverse=True)
        # semantics_log.logger.info(("Use Model: %s" % (os.path.join(model_full_path, filelist[0]))))
        return os.path.join(model_full_path, filelist[0])
    else:
        # semantics_log.logger.info("Model path is not exists")
        print("Model path is not exists")

'''
    读取配置文件，获取打开SocketServer的ip和端口
'''
def getSocketConfig():
    cf = configparser.ConfigParser()
    cf.read("../kdata/config.conf")
    host = str(cf.get("sserver", "host"))
    port = int(cf.get("sserver", "port"))
    return host, port

'''
    获取rabbitmq连接
    :param nodeName 指定配置文件的哪个节点
'''
def getRabbitConn(nodeName):
    cf = configparser.ConfigParser()
    cf.read("../kdata/config.conf")
    host = str(cf.get(nodeName, "host"))
    port = int(cf.get(nodeName, "port"))
    username = str(cf.get(nodeName, "username"))
    password = str(cf.get(nodeName, "password"))
    EXCHANGE_NAME = str(cf.get(nodeName, "EXCHANGE_NAME"))
    vhost = str(cf.get(nodeName, "vhost"))
    routingKey = str(cf.get(nodeName, "routingKey"))

    credentials = pika.PlainCredentials(username=username, password=password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, virtual_host=vhost, credentials=credentials))
    connection.process_data_events()    # 防止主进程长时间等待，而导致rabbitmq主动断开连接，所以要定期发心跳调用
    channel = connection.channel()
    # channel.queue_declare(queue=routingKey, durable=True)    # 定义持久化队列
    # channel.queue_declare(queue=routingKey)  # 定义持久化队列

    return connection, channel, EXCHANGE_NAME, routingKey

backstage_connection, backstage_channel, backstage_EXCHANGE_NAME, backstage_routingKey = getRabbitConn("rabbit2backstage")
# semantics_log.logger.info("rabbit2backstage producer 已启动：%s %s %s %s" % (backstage_connection, backstage_channel, backstage_EXCHANGE_NAME, backstage_routingKey))
print("rabbit2backstage producer 已启动：%s %s %s %s" % (backstage_connection, backstage_channel, backstage_EXCHANGE_NAME, backstage_routingKey))

portrait_connection, portrait_channel, portrait_EXCHANGE_NAME, portrait_routingKey = getRabbitConn("rabbit2portrait")
# semantics_log.logger.info("rabbit2portrait producer 已启动：%s %s %s %s" % (portrait_connection, portrait_channel, portrait_EXCHANGE_NAME, portrait_routingKey))
print("rabbit2portrait producer 已启动：%s %s %s %s" % (portrait_connection, portrait_channel, portrait_EXCHANGE_NAME, portrait_routingKey))


# 手动做心跳机制，避免rabbit server自动断开连接。。自动发心跳机制存在的问题：因rannitmq有流量控制，会屏蔽掉自动心跳机制
def portrait_heartbeat():
    heartbeatDict = {}
    heartbeatDict["daotaiID"] = daotaiID
    heartbeatDict["sentences"] = ""
    heartbeatDict["timestamp"] = str(int(time.time() * 1000))
    heartbeatDict["intention"] = "heartbeat"  # 心跳

    portrait_channel.basic_publish(exchange=portrait_EXCHANGE_NAME,
                                   routing_key=portrait_routingKey,
                                   body=str(heartbeatDict))
    # print("heartbeatDict:", heartbeatDict)
    global timer
    timer = threading.Timer(3, portrait_heartbeat)
    timer.start()


'''
    测试多项式分类器
'''
def test_bayes():
    model_file = "D:/workspace/workspace_python/daotai-semantics/bayes/model/bernousNB/bernousNB_1600674288_9873150105708245_0_None.m"  # 这里写模型文件的绝对路径
    clf = joblib.load(model_file)
    # loadAnswers()    # 加载 意图-答案 表

    sev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP连接
    HOST, PORT = getSocketConfig()
    sev.bind((HOST, PORT))
    sev.listen()
    # semantics_log.logger.info("bayes semantics 已启动。。。")
    print("bayes semantics 已启动。。。")

    conn, addr = sev.accept()    # 这儿会阻塞，等待连接
    # semantics_log.logger.info("%s %s" % (conn, addr))
    print(conn, addr)
    # semantics_log.logger.info((conn, addr))
    sentences = ""
    empty_package_nums = 0    # 记录空包的数量

    while True:
        try:
            recvStr = bytes.decode(conn.recv(4096), encoding='utf-8')
            if len(recvStr) == 0:    # 如果是安卓客户端，当客户端断开时，服务端收到的是空包
                empty_package_nums += 1
                if empty_package_nums >= 200:
                    raise ConnectionResetError
                continue
            else:
                empty_package_nums = 0    # 如果遇到非空包来，则空包数量重新计数

            # print("语音端消息-原始内容：%s" % recvStr)
            # semantics_log.logger.info("语音端消息-原始内容：%s" % recvStr)    # 原始内容保存日志（没说话报的10118错误也会收到并保存）
            # recvJson = eval(recvStr)    # str转dict，这样只能解析单个json的情况，多个json在一个数据包发来会解析失败
            recvJsonArr = resolving_recv(recvStr)    # 同时解析多个传来的json
            # print("============", len(recvJsonArr))

            for recvJson in recvJsonArr:    # 逐个处理每个json
                # print("recvJson: %s" % recvJson)
                # semantics_log.logger.info("recvJson: %s" % recvJson)  # 所有传来的都会记录
                daotaiID = recvJson["daotaiID"]
                sentences = recvJson["sentences"]    # 现在安卓端、语义端、后端都用sentences字段
                timestamp = recvJson["timestamp"]
                msgCalled = recvJson["msgCalled"]  # 被调方：onResult、onError、等等

                if msgCalled == "onBeginOfSpeech":
                    yuyiDict = {}
                    yuyiDict["daotaiID"] = daotaiID
                    yuyiDict["sentences"] = sentences    # “TAG+开始听写”是在onBeginOfSpeech()里面回调的；只有“开始听写”是在startListening()里面调的
                    yuyiDict["timestamp"] = timestamp
                    yuyiDict["intention"] = "onBeginOfSpeech"  # 开始听写

                    # 之后将yuyiDict写入到消息队列
                    backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                    routing_key=backstage_routingKey,
                                                    body=str(yuyiDict))  # 将语义识别结果给到后端
                    print("B1 开始: %s ****** %s" % (sentences, str(getFormatTime(timestamp))))
                elif msgCalled == "onEndOfSpeech":
                    yuyiDict = {}
                    yuyiDict["daotaiID"] = daotaiID
                    yuyiDict["sentences"] = sentences
                    yuyiDict["timestamp"] = timestamp
                    yuyiDict["intention"] = "onEndOfSpeech"  # 停止听写

                    # 之后将yuyiDict写入到消息队列
                    backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                    routing_key=backstage_routingKey,
                                                    body=str(yuyiDict))  # 将语义识别结果给到后端
                    print("B2-1 结束: %s ****** %s" % (sentences, str(getFormatTime(timestamp))))
                elif msgCalled == "onEndOfSpeech_onEvent":
                    yuyiDict = {}
                    yuyiDict["daotaiID"] = daotaiID
                    yuyiDict["sentences"] = sentences
                    yuyiDict["timestamp"] = timestamp
                    yuyiDict["intention"] = "onEndOfSpeech_onEvent"  # 停止听写

                    # 之后将yuyiDict写入到消息队列
                    backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                    routing_key=backstage_routingKey,
                                                    body=str(yuyiDict))  # 将语义识别结果给到后端
                    print("B2-2 结束: %s ****** %s" % (sentences, str(getFormatTime(timestamp))))
                elif msgCalled == "onError":
                    yuyiDict = {}
                    yuyiDict["daotaiID"] = daotaiID
                    yuyiDict["sentences"] = sentences    # TAG+errorCode
                    yuyiDict["timestamp"] = timestamp
                    yuyiDict["intention"] = "onError"    # 报错的信息

                    # 离在线的切换放在语义端进行
                    errorArr = str(sentences).split(" ")
                    if errorArr[0] == "onlineIAT":    # 在线版
                        if errorArr[1] == "11201":    # 说明在线版额度用完了，需要切离线版了
                            # 此处写数据库，指令接收端从库中查询是用在线版还是离线版
                            saveUsed2DB(str(getFormatTime(timestamp)), errorArr[0], 1)

                    # 之后将yuyiDict写入到消息队列（给后端的）
                    backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                    routing_key=backstage_routingKey,
                                                    body=str(yuyiDict))    # 将报错信息给到后端
                    print("B3 报错: %s ****** %s" % (sentences, str(getFormatTime(timestamp))))
                elif msgCalled == "onCloseConn":    # 客户端断开连接
                    # print("%s ****** %s" % (sentences, str(getFormatTime(timestamp))))
                    pass
                elif msgCalled == "onResult":  # 正常获取的识别结果
                    word_list = []
                    new_sentences, railway_dest, shinei_area = get_words(sentences)  # 关键词列表，火车目的地列表，市内目的地列表
                    # bayes_mq_log.logger.info("------------------ onResult start %s ------------------" % str(getFormatTime(int(time.time()))))
                    print("1、------------------ onResult start %s ------------------" % str(getFormatTime(int(time.time()))))
                    if isChat(new_sentences) is False:  # 如果不是咨询类
                        if len(shinei_area) > 0:
                            print("2-1、导航", "-->", shinei_area, "-->", sentences, "-->", str(getFormatTime(timestamp)))
                            # bayes_mq_log.logger.info(("导航", "-->", shinei_area, "-->", sentences, "-->", str(getFormatTime(timestamp))))

                            yuyiDict = {}
                            yuyiDict["daotaiID"] = daotaiID
                            yuyiDict["sentences"] = sentences + "|" + shinei_area[0]
                            yuyiDict["timestamp"] = timestamp
                            yuyiDict["intention"] = "导航"  # 意图

                            # 之后将yuyiDict写入到消息队列
                            backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                            routing_key=backstage_routingKey,
                                                            body=str(yuyiDict))  # 将语义识别结果给到后端
                            print("3-1、yuyiDict: %s" % str(yuyiDict))
                            # bayes_mq_log.logger.info("yuyiDict: %s" % str(yuyiDict))  # 单独写个日志
                            saveYuyi2DB(yuyiDict)

                            # 人物画像端
                            portraitDict = {}  # 人物画像要填的字段
                            portraitDict["source"] = "yuyi"  # 标识来源是语义yuyi端还是backstage后端
                            portraitDict["timestamp"] = timestamp
                            portraitDict["daotaiID"] = daotaiID
                            portraitDict["portrait"] = None  # 画像部分留空
                            portraitDict["savefile"] = ""  # 图片保存路径
                            portraitDict["sentences"] = sentences + "|" + shinei_area[0]  # 询问问题
                            portraitDict["intention"] = "导航"  # 意图
                            portraitDict["intentionLevel"] = "1"  # 意图等级：1级，直接意图；2级，意图的分类；

                            savePortrait2DB(portraitDict)
                            # portrait_channel.basic_publish(exchange=portrait_EXCHANGE_NAME,
                            #                                routing_key=portrait_routingKey,
                            #                                body=str(portraitDict))  # 将语义结果发送到用户画像端
                            # # print("portraitDict: %s" % str(portraitDict))
                            # # bayes_mq_log.logger.info("portraitDict: %s" % str(portraitDict))
                        else:
                            word_list.append(new_sentences)
                            predict = clf.predict(word_list)
                            for left in predict:
                                # if left == "坐车":    # 坐车 不用转为 坐高铁 了，坐高铁 在库中找不到答案
                                #     left = "坐高铁"
                                # answer = getAnswer(left)
                                # thread.start_new_thread(send_msg, ())    # 新开一个线程，通知前端

                                yuyiDict = {}
                                yuyiDict["daotaiID"] = daotaiID

                                if left == "车票查询" or str(sentences.strip()).__contains__("车票查询"):  # 车票查询续上目的地
                                    left = "车票查询"
                                    if len(railway_dest) > 0:
                                        yuyiDict["sentences"] = sentences + "|" + railway_dest[0]
                                    else:
                                        yuyiDict["sentences"] = sentences + "|" + ""  # 如果只识别到“车票查询”而没找到目的地名，直接传空
                                else:
                                    yuyiDict["sentences"] = sentences
                                yuyiDict["timestamp"] = timestamp
                                yuyiDict["intention"] = "听得懂|" + left  # 意图

                                print("2-2、", left, "-->", word_list, "-->", sentences, "-->", str(getFormatTime(timestamp)))
                                # bayes_mq_log.logger.info((left, "-->", word_list, "-->", sentences, "-->", str(getFormatTime(timestamp))))

                                # 之后将yuyiDict写入到消息队列
                                backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                                routing_key=backstage_routingKey,
                                                                body=str(yuyiDict))  # 将语义识别结果给到后端
                                print("3-2、yuyiDict: %s" % str(yuyiDict))
                                # bayes_mq_log.logger.info("yuyiDict: %s" % str(yuyiDict))
                                saveYuyi2DB(yuyiDict)

                                # 人物画像端
                                portraitDict = {}  # 人物画像要填的字段
                                portraitDict["source"] = "yuyi"  # 标识来源是语义yuyi端还是backstage后端
                                portraitDict["timestamp"] = timestamp
                                portraitDict["daotaiID"] = daotaiID
                                portraitDict["portrait"] = None  # 画像部分留空
                                portraitDict["savefile"] = ""  # 图片保存路径

                                if left == "车票查询":  # 车票查询，续上目的地
                                    portraitDict["sentences"] = sentences + "|" + railway_dest  # 询问问题+目的地
                                else:
                                    portraitDict["sentences"] = sentences  # 询问问题
                                portraitDict["intention"] = left  # 意图
                                portraitDict["intentionLevel"] = "1"  # 意图等级：1级，直接意图；2级，意图的分类；

                                savePortrait2DB(portraitDict)
                                # portrait_channel.basic_publish(exchange=portrait_EXCHANGE_NAME,
                                #                                routing_key=portrait_routingKey,
                                #                                body=str(portraitDict))  # 将语义结果发送到用户画像端
                                # # print("portraitDict: %s" % str(portraitDict))
                                # # bayes_mq_log.logger.info("portraitDict: %s" % str(portraitDict))
                    else:
                        print("2-3、咨询类", "-->", sentences, "-->", str(getFormatTime(timestamp)))  # 咨询场景，判断标准：说话字数>5字
                        # bayes_mq_log.logger.info(("咨询类", "-->", sentences, "-->", str(getFormatTime(timestamp))))

                        if str(sentences.strip()).__contains__("转人工"):
                            yuyiDict = {}
                            yuyiDict["daotaiID"] = daotaiID
                            yuyiDict["sentences"] = sentences
                            yuyiDict["timestamp"] = timestamp
                            yuyiDict["intention"] = "artificial"  # 意图

                            # 之后将yuyiDict写入到消息队列
                            backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                            routing_key=backstage_routingKey,
                                                            body=str(yuyiDict))  # 将语义识别结果给到后端
                            print("3-3、yuyiDict: %s" % str(yuyiDict))
                            # bayes_mq_log.logger.info("yuyiDict: %s" % str(yuyiDict))
                            saveYuyi2DB(yuyiDict)

                            # 人物画像端
                            portraitDict = {}  # 人物画像要填的字段
                            portraitDict["source"] = "yuyi"  # 标识来源是语义yuyi端还是backstage后端
                            portraitDict["timestamp"] = timestamp
                            portraitDict["daotaiID"] = daotaiID
                            portraitDict["portrait"] = None  # 画像部分留空
                            portraitDict["savefile"] = ""  # 图片保存路径
                            portraitDict["sentences"] = sentences  # 询问问题
                            portraitDict["intention"] = "artificial"  # 意图
                            portraitDict["intentionLevel"] = "1"  # 意图等级：1级，直接意图；2级，意图的分类；

                            savePortrait2DB(portraitDict)
                            # portrait_channel.basic_publish(exchange=portrait_EXCHANGE_NAME,
                            #                                routing_key=portrait_routingKey,
                            #                                body=str(portraitDict))  # 将语义结果发送到用户画像端
                            # # print("portraitDict: %s" % str(portraitDict))
                            # # bayes_mq_log.logger.info("portraitDict: %s" % str(portraitDict))
                        else:    # 没有出现“转人工”，且听不懂
                            yuyiDict = {}
                            yuyiDict["daotaiID"] = daotaiID
                            yuyiDict["sentences"] = sentences
                            yuyiDict["timestamp"] = timestamp
                            yuyiDict["intention"] = "听不懂"  # 意图

                            # 之后将yuyiDict写入到消息队列
                            backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                            routing_key=backstage_routingKey,
                                                            body=str(yuyiDict))  # 将语义识别结果给到后端
                            print("3-4、yuyiDict: %s" % str(yuyiDict))
                            # bayes_mq_log.logger.info("yuyiDict: %s" % str(yuyiDict))
                            saveYuyi2DB(yuyiDict)

                            # 人物画像端
                            portraitDict = {}  # 人物画像要填的字段
                            portraitDict["source"] = "yuyi"  # 标识来源是语义yuyi端还是backstage后端
                            portraitDict["timestamp"] = timestamp
                            portraitDict["daotaiID"] = daotaiID
                            portraitDict["portrait"] = None  # 画像部分留空
                            portraitDict["savefile"] = ""  # 图片保存路径
                            portraitDict["sentences"] = sentences  # 询问问题
                            portraitDict["intention"] = "artificial"  # 意图
                            portraitDict["intentionLevel"] = "1"  # 意图等级：1级，直接意图；2级，意图的分类；

                            savePortrait2DB(portraitDict)
                            # portrait_channel.basic_publish(exchange=portrait_EXCHANGE_NAME,
                            #                                routing_key=portrait_routingKey,
                            #                                body=str(portraitDict))  # 将语义结果发送到用户画像端
                            # # print("portraitDict: %s" % str(portraitDict))
                            # # bayes_mq_log.logger.info("portraitDict: %s" % str(portraitDict))
                    # bayes_mq_log.logger.info("++++++++++++++++++ onResult end %s ++++++++++++++++++" % str(getFormatTime(int(time.time()))))
                    print("4、++++++++++++++++++ onResult end %s ++++++++++++++++++" % str(getFormatTime(int(time.time()))))
                else:  # 其他情况的处理
                    pass
        except ConnectionResetError as connectionResetError:
            # semantics_log.logger.warn("waiting connect: %s" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            # print("waiting connect: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            conn, addr = sev.accept()
            # semantics_log.logger.info("%s %s" % (conn, addr))
            # print(conn, addr)
            # semantics_log.logger.info((conn, addr))
            continue
        except Exception as e:
            # traceback.print_exc(file=open(semantics_logfile, 'a+'))
            continue


def main():
    heartbeat = threading.Timer(3, portrait_heartbeat)
    heartbeat.start()

    bayes = threading.Thread(target=test_bayes)
    bayes.start()


if __name__ == '__main__':
    main()