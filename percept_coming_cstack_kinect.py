import sys
import time
import socket
import traceback
from utils.pyKinectUtil import Kinect
import numpy as np
from PIL import Image

import face_recognition
from config import *
from utils.commonutil import getFormatTime, crop_face
import configparser
import pika
from utils.CapUtil import Stack
import threading
from keras.utils.data_utils import get_file
from wide_resnet import WideResNet

'''
    来人感知模块
    Note：获取Kinect画面
    Note：自定义Stack(stack_szie)，解决消费速度跟不上生成速度的情况；
    Note：percept_coming.py中也可解决，但在webcam中会报错：[h264 @ 0000000000498f40] error while decoding MB 8 21, bytestream -13
'''

frame_buffer = Stack(30 * 5)
lock = threading.RLock()

def Receive():
    print("start Receive")

    kinect = Kinect()
    while True:
        # color_data = kinect.get_the_data_of_color_depth_infrared_image()  # 获得最新的彩色和深度图像以及红外图像
        color_data = kinect.get_the_data_of_color()    # 只获取最新的色彩图
        if color_data[0] is not None:
            lock.acquire()
            frame_buffer.push(color_data[0])
            lock.release()

def percept():
    nodeName = "rabbit2backstage"  # 读取该节点的数据

    comming_mq_logfile = 'D:/data/daotai_comming_mq.log'
    comming_mq_log = Logger(comming_mq_logfile, level='info')

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
    connection.process_data_events()    # 防止主进程长时间等待，而导致rabbitmq主动断开连接，所以要定期发心跳调用
    backstage_channel = connection.channel()

    # # 创建socket，将来人感知的消息发送到安卓语音识别端
    # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.connect(('192.168.0.144', 50007))

    # 人脸检测
    global face_detect    # 子线程里加载模型，需要将模型指定成全局变量
    face_detect = face_recognition.FaceDetection()  # 初始化mtcnn

    print("face_detect:", face_detect)
    portrait_log.logger.info("face_detect: %s" % (face_detect))

    # 性别年龄识别模型
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"
    global age_gender_model
    age_gender_model = WideResNet(face_size, depth=16, k=8)()
    age_gender_model_dir = os.path.join(os.getcwd(), "model_data").replace("//", "\\")
    fpath = get_file('weights.18-4.06.hdf5',
                     WRN_WEIGHTS_PATH,
                     cache_subdir=age_gender_model_dir)
    age_gender_model.load_weights(fpath)

    while True:
        if int(time.time() * 1000) % 500 == 0:    # 每5s手动发一次心跳，避免rabbit server自动断开连接。自动发心跳机制存在的问题，因rabbitmq有流量控制机制，会屏蔽掉自动心跳机制
            heartbeatDict = {}
            heartbeatDict["daotaiID"] = daotaiID
            heartbeatDict["sentences"] = ""
            heartbeatDict["timestamp"] = str(int(time.time() * 1000))
            heartbeatDict["intention"] = "heartbeat"  # 心跳

            backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                            routing_key=backstage_routingKey,
                                            body=str(heartbeatDict))
            print("heartbeatDict:", heartbeatDict)

        if frame_buffer.size() > 0:
            lock.acquire()
            frame = frame_buffer.pop()    # 每次拿最新的
            frame_buffer.clear()    # 每次拿之后清空缓冲区
            lock.release()

            # print("frame:", type(frame), frame.shape)    # <class 'numpy.ndarray'> (1080, 1920, 3)，（高，宽，通道）
            height, width, channel = frame.shape
            bboxes, landmarks = face_detect.detect_face(frame)
            bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")  # 以高为基准，获得等宽的矩形
            if bboxes == [] or landmarks == []:
                pass
            else:
                print("faces.faceNum:", len(bboxes))
                portrait_log.logger.info("faces.faceNum: %s" % (len(bboxes)))
                box_areas = []
                for i in range(0, len(bboxes)):
                    box = bboxes[i]
                    left, top, right, bottom = box
                    w = right - left
                    h = bottom - top
                    box_areas.append(w * h)    # 人头的面积

                # 找最大的人脸及坐标
                max_face_area = max(box_areas)    # 最大的人脸面积
                max_face_box = bboxes[box_areas.index(max_face_area)]    # 最大人脸面积框对应的坐标
                print("max_face_area: %s, max_face_box: %s" % (max_face_area, max_face_box))
                portrait_log.logger.info("max_face_area: %s, max_face_box: %s" % (max_face_area, max_face_box))
                if max_face_area > face_area_threshold:
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

                    left, top, right, bottom = max_face_box

                    commingDict = {}
                    commingDict["daotaiID"] = daotaiID
                    commingDict["sentences"] = "%s,%s,%s,%s,%s,%s,%s,%s,%s" % (gender, str(age), str(left), str(top), str(right), str(bottom), str(face_area_threshold), str(height), str(width))    # sentences字段填性别、年龄、位置（左上右下），逗号隔开
                    commingDict["timestamp"] = str(int(time.time() * 1000))
                    commingDict["intention"] = "mycoming"  # 表示有人来了

                    # # 将来人消息发送到语音端
                    # client.send(str(commingDict).encode('utf-8'))  # 收发消息一定要二进制，记得编码

                    print("commingDict: %s" % (commingDict))
                    portrait_log.logger.info("commingDict: %s" % (commingDict))
                    try:
                        backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                        routing_key=backstage_routingKey,
                                                        body=str(commingDict))  # 将语义识别结果给到后端
                    except ConnectionResetError as e:
                        comming_mq_log.logger.error("ConnectionResetError: %s", traceback.format_exc())
                        credentials = pika.PlainCredentials(username=username, password=password)
                        connection = pika.BlockingConnection(
                            pika.ConnectionParameters(host=host, port=port, virtual_host=vhost,
                                                      credentials=credentials))
                        connection.process_data_events()  # 防止主进程长时间等待，而导致rabbitmq主动断开连接，所以要定期发心跳调用
                        backstage_channel = connection.channel()

                        comming_mq_log.logger.info("rabbit2portrait producer 已重连：%s %s %s %s" % (
                            connection, backstage_channel, backstage_EXCHANGE_NAME, backstage_routingKey))
                        print("rabbit2portrait producer 已重连：%s %s %s %s" % (
                            connection, backstage_channel, backstage_EXCHANGE_NAME, backstage_routingKey))

                        # 重连后再发一次
                        backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                        routing_key=backstage_routingKey,
                                                        body=str(commingDict))  # 将语义识别结果给到后端
                    print("已写入消息队列-commingDict: %s" % str(commingDict))
                    comming_mq_log.logger.info("已写入消息队列-commingDict: %s" % str(commingDict))
                    # time.sleep(3)  # 识别到有人来了，等人问完问题再进行识别

            if height != 480 or width != 640:
                frame = cv2.resize(frame, (640, 480))    # resize时的顺序为：宽，高
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=percept)
    p1.start()
    time.sleep(5)
    p2.start()