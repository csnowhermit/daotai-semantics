import os
import cv2
import pika
import json
import time
import datetime
import traceback
import configparser
from PIL import Image
import numpy as np
import argparse
from yolo import YOLO
from config import *
from mymodel import *
from utils.commonutil import getFormatTime

'''
    portrait端，作为rabbitmq消费端口，接收语义端和后端的消息，进行人物画像识别，并入库
'''

# 日志文件
portrait_log = Logger('D:/data/daotai_portrait.log', level='info')

'''
    获取摄像头连接
'''
def getCap(input_webcam):
    if input_webcam == "0":
        input_webcam = int(0)

    cap_retry = 0
    while True:
        cap = cv2.VideoCapture(input_webcam, cv2.CAP_DSHOW)
        # cap = cv2.VideoCapture(input_webcam)
        if cap.isOpened():
            return cap
        else:
            if cap_retry < 5:
                cap_retry += 1
                time.sleep(0.5 * cap_retry)
                portrait_log.logger.error("摄像头打开失败，正在第 %d 次重试" % (cap_retry))

'''
    从普通摄像机抓取图片
'''
def captureImage(input_webcam):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_count = 0

    print("face_detect:", face_detect)
    portrait_log.logger.info("face_detect: %s" % (face_detect))

    cap = getCap(input_webcam)

    start_time = time.time()
    retry = 0    # 读cap.read()重试次数
    while True:
        ret, frame = cap.read()

        # print("===== frame.shape =====", frame.shape)
        if (frame_count % frame_interval) == 0:  # 跳帧处理，解决算法和采集速度不匹配
            # if frame_count > -1:
            # frame = np.asanyarray(frame)    # 本身frame就是np.ndarray，不用再转
            if frame is None:
                retry += 1
                time.sleep(1)  # 读取失败后立马重试没有任何意义
                continue
            if retry > 5:
                break

            # print("frame:", type(frame), frame.shape)    # <class 'numpy.ndarray'> (480, 640, 3)，（高，宽，通道）
            bboxes, landmarks = face_detect.detect_face(frame)
            bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")  # 以高为基准，获得等宽的矩形
            if bboxes == [] or landmarks == []:
                pass
            else:
                print("faces.faceNum:", len(bboxes))
                portrait_log.logger.info("faces.faceNum: %s" % (len(bboxes)))
                for i in range(0, len(bboxes)):
                    box = bboxes[i]
                    left, top, right, bottom = box
                    w = right - left
                    h = bottom - top
                    if w*h > face_area_threshold:
                        # print("mtcnn-bboxes--> ", bboxes)
                        # print("mtcnn-landmarks--> ", landmarks)
                        return frame, bboxes, landmarks
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
    cap.release()
    cv2.destroyAllWindows()
    return None, [], []

'''
    从kinect获取图片
'''
def captureImageFromKinect(kinect):
    retry = 0    # 抓图出错重试次数

    print("face_detect:", face_detect)
    portrait_log.logger.info("face_detect: %s" % (face_detect))

    while True:
        # color_data = kinect.get_the_data_of_color_depth_infrared_image()  # 获得最新的彩色和深度图像以及红外图像
        color_data = kinect.get_the_data_of_color()    # 只获取最新的色彩图
        frame = color_data[0]
        if frame is not None:
            # print("frame:", type(frame), frame.shape)    # <class 'numpy.ndarray'> (480, 640, 3)，（高，宽，通道）
            bboxes, landmarks = face_detect.detect_face(frame)
            bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")  # 以高为基准，获得等宽的矩形
            if bboxes == [] or landmarks == []:
                pass
            else:
                box_areas = []    # 人头面积列表
                print("faces.faceNum:", len(bboxes))
                portrait_log.logger.info("faces.faceNum: %s" % (len(bboxes)))
                for i in range(0, len(bboxes)):
                    box = bboxes[i]
                    left, top, right, bottom = box
                    w = right - left
                    h = bottom - top
                    box_areas.append(w * h)  # 人头的面积

                # 找最大的人脸及坐标
                max_face_area = max(box_areas)  # 最大的人脸面积
                max_face_box = bboxes[box_areas.index(max_face_area)]  # 最大人脸面积框对应的坐标
                if max_face_area > face_area_threshold:
                    # print("mtcnn-bboxes--> ", bboxes)
                    # print("mtcnn-landmarks--> ", landmarks)
                    return frame, bboxes, landmarks
        else:
            retry += 1
            time.sleep(0.05)
            if retry > 50:
                break
    return None, [], []

'''
    检测，生成人物画像
    :param frame 要求传入的图片格式：<class 'numpy.ndarray'>
    :param gray_image frame的灰度图，np.ndarray
    :param bbox 第一步检测到的人脸框
    :param landmarks 第一次检测到的人脸关键点
'''
def detect_portrait(yolo, frame, gray_image, bbox, landmarks):
    image = Image.fromarray(frame)    # 转成 PIL.Image

    image, featureDict = yolo.detect_image(image, gray_image, bbox)    # 行人及行李检测

    result = np.asarray(image)    # 又转回 np.ndarray

    return result, featureDict
    # cv2.imshow("r", result)
    # cv2.waitKey(1)

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
    connection.process_data_events()    # 防止主进程长时间等待，而导致rabbitmq主动断开连接，所以要定期发心跳调用
    channel = connection.channel()
    channel.exchange_declare(exchange=EXCHANGE_NAME,
                             exchange_type='direct')    # 声明交换机
    channel.queue_declare(queue=queueName)    # 声明队列。消费者需要这样代码，生产者不需要
    channel.queue_bind(queue=queueName, exchange=EXCHANGE_NAME, routing_key=routingKey)    # 绑定队列和交换机

    return channel, EXCHANGE_NAME, queueName, routingKey


# 定义一个回调函数来处理，这边的回调函数就是将信息打印出来。
def callback(ch, method, properties, body):
    print("++++++++++ start a portrait detect ++++++++++ %s" % (getFormatTime(str(int(time.time())))))
    portrait_log.logger.info("++++++++++ start a portrait detect ++++++++++ %s" % (getFormatTime(str(int(time.time())))))
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
        formatted_time = getFormatTime(int(timestamp))  # 得到格式化后的时间

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

        print("complete-portrait: %s" % (portraitDict))
        portrait_log.logger.info("complete-portrait: %s" % (portraitDict))  # 写日志也行，入库也行
    except Exception as e:
        portrait_log.logger.error(traceback.format_exc())
    print("========== end a portrait detect ========== %s" % (getFormatTime(str(int(time.time())))))
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