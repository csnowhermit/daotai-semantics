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
from utils.commonutil import getFormatTime, crop_face
import configparser
import pika
from portrait_detect import getCap

'''
    来人感知模块
'''

if __name__ == '__main__':
    nodeName = "rabbit2backstage"    # 读取该节点的数据

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

    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_count = 0

    print("face_detect:", face_detect)
    portrait_log.logger.info("face_detect: %s" % (face_detect))

    cap = getCap(input_webcam)    # cv2.VideoCapture(0, cv2.CAP_DSHOW)
    portrait_log.logger.info("cap.isOpened(): %s %s" % (cap.isOpened(), input_webcam))

    start_time = time.time()
    retry = 0  # 读cap.read()重试次数
    while True:
        ret, frame = cap.read()

        if (frame_count % frame_interval) == 0:  # 跳帧处理，解决算法和采集速度不匹配
            # if frame_count > -1:
            # frame = np.asanyarray(frame)
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
                    if w * h > face_area_threshold:
                        # print("mtcnn-bboxes--> ", bboxes)
                        # print("mtcnn-landmarks--> ", landmarks)

                        # 这里新增来人的性别年龄识别
                        image = Image.fromarray(frame)

                        # 2.性别年龄检测
                        tmp = crop_face(image, box, margin=40,
                                        size=face_size)  # 裁剪脑袋部分，并resize，image：<class 'PIL.Image.Image'>
                        faces = [[left, top, right, bottom]]  # 做成需要的格式：[[], [], []]
                        face_imgs = np.empty((len(faces), face_size, face_size, 3))
                        # face_imgs[0, :, :, :] = cv2.resize(np.asarray(tmp), (face_size, face_size))    # PIL.Image转为np.ndarray，不resize会报错：ValueError: could not broadcast input array from shape (165,165,3) into shape (64,64,3)
                        face_imgs[0, :, :, :] = tmp
                        print("face_imgs:", type(face_imgs), face_imgs.shape)

                        results = age_gender_model.predict(face_imgs)  # 性别年龄识别
                        predicted_genders = results[0]
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()

                        gender = "F" if predicted_genders[0][0] > 0.5 else "M"
                        age = int(predicted_ages[0])

                        commingDict = {}
                        commingDict["daotaiID"] = daotaiID
                        commingDict["sentences"] = gender + "," + str(age)  # sentences字段填性别和年龄，逗号隔开
                        commingDict["timestamp"] = str(int(time.time()) * 1000)
                        commingDict["intention"] = "mycoming"  # 表示有人来了

                        print("commingDict: %s" % (commingDict))
                        portrait_log.logger.info("commingDict: %s" % (commingDict))
                        backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                        routing_key=backstage_routingKey,
                                                        body=str(commingDict))  # 将语义识别结果给到后端
                        time.sleep(3)  # 识别到有人来了，等人问完问题再进行识别
            cv2.imshow("coming", frame)
            cv2.waitKey(1)
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
    cap.release()
    cv2.destroyAllWindows()