import sys
import time
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
# from mymodel import face_detect, age_gender_model

'''
    来人感知模块
    Note：自定义Stack(stack_szie)，解决消费速度跟不上生成速度的情况；
    Note：percept_coming.py中也可解决，但在webcam中会报错：[h264 @ 0000000000498f40] error while decoding MB 8 21, bytestream -13
'''

frame_buffer = Stack(30 * 5)
lock = threading.RLock()

def Receive():
    print("start Receive")
    cap = cv2.VideoCapture(0)
    # cap = getCap(input_webcam)
    print("cap.isOpened(): %s %s" % (cap.isOpened(), input_webcam))
    portrait_log.logger.info("cap.isOpened(): %s %s" % (cap.isOpened(), input_webcam))
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_count = 0

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if ret is True:
            lock.acquire()
            frame_buffer.push(frame)
            lock.release()
        # if (frame_count % frame_interval) == 0:  # 跳帧处理，解决算法和采集速度不匹配
        #     if ret is True:
        #         # q.put(frame)
        #         lock.acquire()
        #         frame_buffer.push(frame)
        #         lock.release()
        #     # Check our current fps
        #     end_time = time.time()
        #     if (end_time - start_time) > fps_display_interval:
        #         frame_rate = int(frame_count / (end_time - start_time))
        #         start_time = time.time()
        #         frame_count = 0

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
    backstage_channel = connection.channel()

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
        if frame_buffer.size() > 0:
            lock.acquire()
            frame = frame_buffer.pop()    # 每次拿最新的
            frame_buffer.clear()    # 每次拿之后清空缓冲区
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
                    print("face_area:", w * h)
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
                        commingDict["message"] = gender + "," + str(age)  # sentences字段填性别和年龄，逗号隔开
                        commingDict["timestamp"] = str(int(time.time()) * 1000)
                        commingDict["intention"] = "mycoming"  # 表示有人来了

                        print("commingDict: %s" % (commingDict))
                        portrait_log.logger.info("commingDict: %s" % (commingDict))
                        backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                                        routing_key=backstage_routingKey,
                                                        body=str(commingDict))  # 将语义识别结果给到后端
                        print("已写入消息队列-commingDict: %s" % str(commingDict))
                        comming_mq_log.logger.info("已写入消息队列-commingDict: %s" % str(commingDict))
                        time.sleep(3)  # 识别到有人来了，等人问完问题再进行识别
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=percept)
    p1.start()
    time.sleep(5)
    p2.start()
