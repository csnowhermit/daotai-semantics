import os
import cv2
import pika
import json
import time
import datetime
import traceback
import configparser
import argparse
from yolo import YOLO
from config import *
from mymodel import *
from utils.commonutil import getFormatTime
from portrait_detect import captureImage, detect_portrait

'''
    portrait端，作为rabbitmq消费端口，接收语义端和后端的消息，进行人物画像识别，并入库
'''

def getRabbitConn(nodeName):
    cf = configparser.ConfigParser()
    cf.read("./kdata/config.conf")
    host = str(cf.get(nodeName, "host"))
    port = int(cf.get(nodeName, "port"))
    username = str(cf.get(nodeName, "username"))
    password = str(cf.get(nodeName, "password"))
    EXCHANGE_NAME = str(cf.get(nodeName, "EXCHANGE_NAME"))
    vhost = str(cf.get(nodeName, "vhost"))
    routingKey = str(cf.get(nodeName, "routingKey"))
    queueName = str(cf.get(nodeName, "QUEUE_NAME"))

    credentials = pika.PlainCredentials(username=username, password=password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, virtual_host=vhost, credentials=credentials))
    channel = connection.channel()
    channel.exchange_declare(exchange=EXCHANGE_NAME,
                             exchange_type='direct')    # 声明交换机
    channel.queue_declare(queue=queueName)    # 声明队列。消费者需要这样代码，生产者不需要
    channel.queue_bind(queue=queueName, exchange=EXCHANGE_NAME, routing_key=routingKey)    # 绑定队列和交换机

    return channel, EXCHANGE_NAME, queueName, routingKey


# 定义一个回调函数来处理，这边的回调函数就是将信息打印出来。
def callback(ch, method, properties, body):
    portrait_log.logger.info("********** start a portrait detect ********** %s" % (getFormatTime(str(int(time.time())))))
    print(" [x] Received %r" % body)
    portrait_log.logger.info(" [x] Received %r" % body)
    try:
        recvStr = str(body, encoding="utf-8")
        # print("recvStr: ", recvStr)
        # recvJson = json.loads(recvStr)  # 接收到的json，包含字段：
        # portrait_log.logger.info(recvJson)

        recvDict = eval(recvStr)    # str转dict
        source = recvDict["source"]
        daotaiID = recvDict["daotaiID"]
        timestamp = recvDict["timestamp"]
        formatted_time = getFormatTime(timestamp)  # 得到格式化后的时间

        if os.path.exists(portrait_img_path) is False:
            os.makedirs(portrait_img_path)
        savefile = os.path.join(portrait_img_path, "%s_%s.jpg" % (daotaiID, formatted_time))


        frame, facebboxes, landmarks = captureImage(input_webcam)
        if frame is not None:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("frame:", type(frame), frame.shape)
            result, featureDict = detect_portrait(yolo=yolo, frame=frame, gray_image=gray_image, bbox=facebboxes,
                                                  landmarks=landmarks)  # 应用中调这行代码即可
            cv2.imwrite(savefile, result)
        else:
            result = frame
            savefile = ""
            featureDict = {"errorMsg": "暂未捕捉到画面"}
        # 拼接最后的json
        portraitDict = {}
        portraitDict["source"] = source  # 标识来源是语义yuyi端还是backstage后端
        portraitDict["timestamp"] = timestamp
        portraitDict["daotaiID"] = daotaiID
        portraitDict["portrait"] = featureDict  # 行李、性别、年龄、表情
        portraitDict["savefile"] = savefile  # 图片保存路径
        portraitDict["sentences"] = recvDict["sentences"]  # 询问问题
        portraitDict["intention"] = recvDict["intention"]  # 意图
        portraitDict["intentionLevel"] = recvDict["intentionLevel"]    # 意图级别

        portrait_log.logger.info("complete-portrait: %s" % portraitDict)  # 写日志也行，入库也行
    except Exception as e:
        portrait_log.logger.error(traceback.format_exc())
    portrait_log.logger.info("========== end a portrait detect ========== %s" % (getFormatTime(str(int(time.time())))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    FLAGS = parser.parse_args()
    yolo = YOLO(**vars(FLAGS))

    consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey = getRabbitConn("rabbit2portrait")
    portrait_log.logger.info("rabbit consumer2portrait 已启动：%s %s %s %s" % (consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey))
    print("rabbit consumer2portrait 已启动：%s %s %s %s" % (consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey))

    consumer_channel.basic_consume(queue=consumer_queueName, on_message_callback=callback, auto_ack=True)    # 这里写的是QUEUE_NAME，而不是routingKey

    print(' [*] Waiting for messages. To exit press CTRL+C')

    # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理。按ctrl+c退出。
    consumer_channel.start_consuming()