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

'''
    来人感知模块
'''

if __name__ == '__main__':
    nodeName = "rabbit2backstage"    # 读取该节点的数据

    cf = configparser.ConfigParser()
    cf.read("../kdata/config.conf")
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

    if input_webcam == "0":
        input_webcam = int(0)

    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_count = 0

    print("face_detect:", face_detect)
    portrait_log.logger.info("face_detect: %s" % (face_detect))

    cap = cv2.VideoCapture(input_webcam)
    start_time = time.time()
    retry = 0  # 读cap.read()重试次数
    while True:
        ret, frame = cap.read()

        if frame is None:
            retry += 1
            time.sleep(retry * 2)  # 读取失败后立马重试没有任何意义
            if retry > 10:
                break
        if (frame_count % frame_interval) == 0:  # 跳帧处理，解决算法和采集速度不匹配
            # if frame_count > -1:
            frame = np.asanyarray(frame)

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
                        commingDict["intention"] = "mycoming"    # 表示有人来了

                        backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                        routing_key=backstage_routingKey,
                                                        body=str(commingDict))  # 将语义识别结果给到后端
                        time.sleep(3)    # 识别到有人来了，等人问完问题再进行识别

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
    cap.release()
    cv2.destroyAllWindows()

