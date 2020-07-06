import sys
import time
import argparse
import cv2
import numpy as np
import traceback
from yolo import YOLO
from PIL import Image
import timer
import face_recognition
from config import *
from mymodel import *
from utils.commonutil import getFormatTime
import configparser
import pika
from utils.CapUtil import Stack
from portrait_detect import getCap
import threading

'''
    来人感知模块
    Note：自定义Stack(stack_szie)，解决消费速度跟不上生成速度的情况；
    Note：percept_coming.py中也可解决，但在webcam中会报错：[h264 @ 0000000000498f40] error while decoding MB 8 21, bytestream -13
'''

frame_buffer = Stack(30 * 5)
lock = threading.RLock()

def Receive():
    print("start Receive")
    print("start Receive")
    cap = cv2.VideoCapture(0)
    # cap = getCap(input_webcam)
    portrait_log.logger.info("cap.isOpened(): %s %s" % (cap.isOpened(), input_webcam))
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_count = 0

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if (frame_count % frame_interval) == 0:  # 跳帧处理，解决算法和采集速度不匹配
            if ret is True:
                # q.put(frame)
                lock.acquire()
                frame_buffer.push(frame)
                lock.release()
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

def percept():
    nodeName = "rabbit2backstage"  # 读取该节点的数据

    cf = configparser.ConfigParser()
    cf.read("./kdata/config.conf")
    host = str(cf.get(nodeName, "host"))
    port = int(cf.get(nodeName, "port"))
    username = str(cf.get(nodeName, "username"))
    password = str(cf.get(nodeName, "password"))
    backstage_EXCHANGE_NAME = str(cf.get(nodeName, "EXCHANGE_NAME"))
    vhost = str(cf.get(nodeName, "vhost"))
    backstage_routingKey = str(cf.get(nodeName, "routingKey"))

    credentials = pika.PlainCredentials(username=username, password=password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=host, port=port, virtual_host=vhost, credentials=credentials))
    backstage_channel = connection.channel()

    print("face_detect:", face_detect)
    portrait_log.logger.info("face_detect: %s" % (face_detect))

    while True:
        if frame_buffer.size() > 0:
            lock.acquire()
            frame = frame_buffer.pop()    # 每次拿最新的
            lock.release()

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
                    if w * h > face_area_threshold:
                        # print("mtcnn-bboxes--> ", bboxes)
                        # print("mtcnn-landmarks--> ", landmarks)
                        commingDict = {}
                        commingDict["daotaiID"] = daotaiID
                        commingDict["message"] = ""
                        commingDict["timestamp"] = str(int(time.time()) * 1000)
                        commingDict["intention"] = "mycoming"  # 表示有人来了

                        print("commingDict: %s" % (commingDict))
                        portrait_log.logger.info("commingDict: %s" % (commingDict))
                        backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                        routing_key=backstage_routingKey,
                                                        body=str(commingDict))  # 将语义识别结果给到后端
                        # time.sleep(3)  # 识别到有人来了，等人问完问题再进行识别
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=percept)
    p1.start()
    time.sleep(5)
    p2.start()
