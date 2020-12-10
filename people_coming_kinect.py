#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import argparse
from timeit import time
import warnings
import cv2
import torch
from keras.models import load_model
import traceback
import hashlib
import numpy as np

import face_recognition
from PIL import Image, ImageDraw, ImageFont
import colorsys
import threading
import configparser
import pika

from detector import build_detector
from deep_sort import build_tracker
from config import daotaiID, face_size, expand_multiple, gender_ratio_threshold, emotion_labels, emotion_offsets, face_area_threshold
from common.trackUtil import getUsefulTrack
from common.Stack import Stack
from common.cleanUtil import cleaning_box
from common.ImageUtil import crop_face
from common.commonUtil import bindFaceAndPerson, isin
from utils.parser import get_config
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from utils.dbUtil import saveMyComing2DB, savePortrait2DB
from utils.dateUtil import formatTimestamp
from utils.pyKinectUtil import Kinect
from wide_resnet import WideResNet


warnings.filterwarnings('ignore')

# 发送来人消息
def send_comming(commingDict):
    # 读取配置文件并创建rabbit producer
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
    backstage_queueName = str(cf.get(nodeName, "QUEUE_NAME"))

    credentials = pika.PlainCredentials(username=username, password=password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=host, port=port, heartbeat=0, virtual_host=vhost, credentials=credentials))
    connection.process_data_events()  # 防止主进程长时间等待，而导致rabbitmq主动断开连接，所以要定期发心跳调用
    backstage_channel = connection.channel()

    # 以下原来是在消费者里面
    backstage_channel.exchange_declare(exchange=backstage_EXCHANGE_NAME,
                                       exchange_type='direct')  # 声明交换机
    backstage_channel.queue_declare(queue=backstage_queueName)  # 声明队列。消费者需要这样代码，生产者不需要
    backstage_channel.queue_bind(queue=backstage_queueName, exchange=backstage_EXCHANGE_NAME,
                                 routing_key=backstage_routingKey)  # 绑定队列和交换机

    backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                    routing_key=backstage_routingKey,
                                    body=str(commingDict))  # 将语义识别结果给到后端
    connection.close()


'''
    视频流读取线程：读取到自定义缓冲区
'''
def capture_thread(input_webcam, frame_buffer, lock):
    print("start Receive")

    kinect = Kinect()
    while True:
        # color_data = kinect.get_the_data_of_color_depth_infrared_image()  # 获得最新的彩色和深度图像以及红外图像
        color_data = kinect.get_the_data_of_color()  # 只获取最新的色彩图
        if color_data[0] is not None:
            lock.acquire()
            frame_buffer.push(color_data[0])
            lock.release()


def detect_thread(cfg, frame_buffer, lock):
    use_cuda = torch.cuda.is_available()    # 是否用cuda

    # 人脸检测
    global face_detect  # 子线程里加载模型，需要将模型指定成全局变量
    face_detect = face_recognition.FaceDetection()  # 初始化mtcnn

    print("face_detect:", face_detect)
    # comming_log.logger.info("face_detect: %s" % (face_detect))

    # 性别年龄识别模型
    global age_gender_model
    age_gender_model = WideResNet(face_size, depth=16, k=8)()
    age_gender_model.load_weights("./model_data/weights.18-4.06.hdf5")

    # 表情识别
    global emotion_classifier
    emotion_model_path = "./model_data/fer2013_mini_XCEPTION.102-0.66.hdf5"
    emotion_classifier = load_model(emotion_model_path, compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]

    detector = build_detector(cfg, use_cuda=use_cuda)  # 构建检测器
    deepsort = build_tracker(cfg, use_cuda=use_cuda, n_start=0)     # 构建追踪器
    class_names = detector.class_names

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    image_size = (640, 480)
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image_size[1] + 0.5).astype('int32'))  # 640*480
    thickness = (image_size[0] + image_size[1]) // 300

    while True:
        try:
            if frame_buffer.size() > 0:
                lock.acquire()
                frame = frame_buffer.pop()  # 每次拿最新的
                frame_buffer.clear()        # 拿完之后清空缓冲区，避免短期采集线程拿不到数据帧而导致识别线程倒退识别
                lock.release()

                curr_timestamp = time.time()
                frame = np.array(frame)
                # 先做检测
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # BGR转RGB，用于识别。这里转换移到detector()里做
                height, width, channel = frame.shape
                bbox_xyxy, cls_conf, cls_ids = detector(frame)  # 这里所有的检出，box格式均为：左上右下

                # 经过对原始框的预处理，现在：person：左上宽高；物品：上左下右
                (person_classes, person_boxs, person_scores), \
                (other_classes, other_boxs, other_scores) = cleaning_box(bbox_xyxy, cls_conf, cls_ids, class_names)

                # 再做追踪
                deepsort.update(frame, person_classes, person_boxs, person_scores)

                # 这里出现bug：误检，只检出一个人，为什么tracker.tracks中有三个人
                # 原因：人走了，框还在
                # 解决办法：更新后的tracker.tracks与person_boxs再做一次iou，对于每个person_boxs，只保留与其最大iou的track

                trackList = getUsefulTrack(person_boxs, deepsort.tracker.tracks)

                # 再做人脸检测
                face_bboxes, face_landmarks = face_detect.detect_face(frame)
                face_bboxes, face_landmarks = face_detect.get_square_bboxes(face_bboxes, face_landmarks, fixed="height")  # 以高为基准，获得等宽的矩形

                personList = [[track.classes, track.track_id, track.to_tlbr(), track.score, track.state] for track in trackList]    # 人体列表：左上右下
                # 绑定人脸和人体的关系
                askPersonList = bindFaceAndPerson(face_bboxes, personList)  # （人体框-上左下右，人体面积，人脸框-左上右下，人脸面积）

                # 提取具体提问人的性别，年龄，表情信息
                detail_info = []    # 按顺序：性别，年龄，表情
                my_coming = False    # 默认没人来
                if len(askPersonList) > 0:
                    result = sorted(askPersonList, key=lambda x: x[5], reverse=True)  # reverse=True表示降序
                    asker = result[0]  # 提问人

                    for person in askPersonList:
                        if asker == person:
                            this_person_type = "asker"
                            my_coming = True
                        else:
                            this_person_type = "other"
                        p_class, person_id, person_box, person_box_area, face_box, face_box_area, score, track_state = person    # 类别，id，人物框，人体框面积，人脸框，人脸面积，得分，追踪状态
                        person_top, person_left, person_bottom, person_right = person_box    # 上左下右

                        w = person_right - person_left
                        h = person_bottom - person_top
                        personNewBox = (person_top - expand_multiple * h, person_left - expand_multiple * w,
                                        person_bottom + expand_multiple * h,
                                        person_right + expand_multiple * w)  # 询问人的新框：行李检测有小框

                        # 1.行李检测
                        lugList = []
                        for i in range(len(other_classes)):  # 找行李，人物框扩大一倍，行李中心点在框内的，就算是携带的行李
                            p_class, box, score = other_classes[i], other_boxs[i], other_scores[i]
                            lug_top, lug_left, lug_bottom, lug_right = box
                            w = lug_right - lug_left
                            h = lug_bottom - lug_top
                            luggageCenter = (lug_left + w / 2, lug_top + h / 2)  # 行李的中心点
                            tag = isin(luggageCenter, personNewBox)
                            if tag is True:
                                lugList.append((p_class, box, score))

                        # 2.性别年龄检测
                        face_left, face_top, face_right, face_bottom = face_box  # 左上右下
                        image = Image.fromarray(frame)
                        tmp = crop_face(image, face_box, margin=40,
                                        size=face_size)  # 裁剪脑袋部分，并resize，image：<class 'PIL.Image.Image'>
                        faces = [[face_left, face_top, face_right, face_bottom]]  # 做成需要的格式：[[], [], []]
                        face_imgs = np.empty((len(faces), face_size, face_size, 3))
                        # face_imgs[0, :, :, :] = cv2.resize(np.asarray(tmp), (face_size, face_size))    # PIL.Image转为np.ndarray，不resize会报错：ValueError: could not broadcast input array from shape (165,165,3) into shape (64,64,3)
                        face_imgs[0, :, :, :] = tmp

                        results = age_gender_model.predict(face_imgs)  # 性别年龄识别
                        predicted_genders = results[0]
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()

                        gender = "F" if predicted_genders[0][0] > 0.5 else "M"  # 性别初筛
                        gender_ratio = max(predicted_genders[0])  # 性别概率

                        if gender_ratio < gender_ratio_threshold:  # 低于性别阀值，直接说 乘客
                            gender = "O"

                        age = int(predicted_ages[0])

                        # print("1、asker_info:", asker_info)

                        # 3.表情识别
                        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        x1, x2, y1, y2 = apply_offsets(face_box, emotion_offsets)  # 左右上下
                        # print("emotion reco: ", type(gray_image), x1, x2, y1, y2)
                        gray_face = gray_image[y1:y2, x1:x2]
                        try:
                            gray_face = cv2.resize(gray_face, (emotion_target_size))
                        except:
                            pass

                        gray_face = preprocess_input(gray_face, True)
                        gray_face = np.expand_dims(gray_face, 0)
                        gray_face = np.expand_dims(gray_face, -1)
                        emotion_prediction = emotion_classifier.predict(gray_face)
                        emotion_probability = np.max(emotion_prediction)
                        emotion_label_arg = np.argmax(emotion_prediction)
                        emotion_text = emotion_labels[emotion_label_arg]
                        # asker_info.append(emotion_text)
                        detail_info.append((this_person_type, lugList, face_box, gender, age, emotion_text))

                        # 4.其他。。。后续待加
                        # ====================
                savefile = "D:/daotai/" + formatTimestamp(curr_timestamp, format="%Y%m%d_%H%M%S", ms=True) + ".jpg"
                # 向后端传送来人的消息
                for person, info in zip(askPersonList, detail_info):
                    p_class, person_id, person_box, person_box_area, face_box, face_box_area, score, track_state = person  # 类别，id，人物框，人体框面积，人脸框，人脸面积
                    this_person_type, lugList, face_box, gender, age, emotion_text = info
                    left, top, right, bottom = face_box

                    if this_person_type == "asker":    # 询问者
                        # 来人感知端
                        commingDict = {}
                        commingDict["daotaiID"] = daotaiID
                        commingDict["sentences"] = "%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
                                gender, str(age), str(left), str(top), str(right), str(bottom),
                                str(face_area_threshold),
                                str(height), str(width))  # sentences字段填性别、年龄、位置（左上右下），逗号隔开
                        commingDict["timestamp"] = str(int(time.time() * 1000))
                        commingDict["intention"] = "mycoming"  # 表示有人来了

                        # print("commingDict: %s" % (commingDict))
                        # comming_log.logger.info("commingDict: %s" % (commingDict))
                        send_comming(str(commingDict))
                        print("已写入消息队列-commingDict: %s" % str(commingDict))
                        # comming_mq_log.logger.info("已写入消息队列-commingDict: %s" % str(commingDict))
                        saveMyComing2DB(commingDict)

                        # 用户画像端
                        featureDict = {}  # 人物画像的项点和值
                        featureDict["luggage"] = lugList
                        featureDict["gender"] = gender
                        featureDict["age"] = age
                        featureDict["emotion"] = emotion_text  # 表情

                        # 拼接最后的json
                        portraitDict = {}
                        portraitDict["savefile"] = savefile if my_coming is True else ""
                        portraitDict["timestamp"] = formatTimestamp(curr_timestamp, format="%Y-%m-%d_%H:%M:%S",
                                                                    ms=True)
                        portraitDict["daotaiID"] = daotaiID
                        portraitDict["portrait"] = featureDict  # 行李、性别、年龄、表情

                        print("complete-portrait: %s" % (portraitDict))
                        savePortrait2DB(portraitDict)
                # 准备做标记
                image = Image.fromarray(frame[..., ::-1])  # bgr to rgb，转成RGB格式进行做标注
                draw = ImageDraw.Draw(image)

                for person, info in zip(askPersonList, detail_info):
                    p_class, person_id, person_box, person_box_area, face_box, face_box_area, score, track_state = person  # 类别，id，人物框，人体框面积，人脸框，人脸面积
                    this_person_type, lugList, face_box, gender, age, emotion_text = info

                    label = '{} {:.2f} {} {} {} {} {}'.format(this_person_type, score, person_id, track_state, gender, age, emotion_text)
                    label_size = draw.textsize(label, font)

                    left, top, right, bottom = person_box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                    # print(label, (left, top), (right, bottom))

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=colors[class_names.index(p_class)])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=colors[class_names.index(p_class)])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
                result = np.asarray(image)  # 这时转成np.ndarray后是rgb模式，out.write(result)保存为视频用
                # bgr = rgb[..., ::-1]    # rgb转bgr
                result = result[..., ::-1]  # 转成BGR做cv2.imwrite()用

                if my_coming is True:
                    cv2.imwrite(savefile, result)
                if height != 480 or width != 640:
                    result = cv2.resize(result, (640, 480))  # resize时的顺序为：宽，高
                cv2.imshow("frame", result)
                cv2.waitKey(1)
        except Exception as e:
            traceback.print_exc()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov5s.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

if __name__ == '__main__':
    # 配置项
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    # 自定义识别缓冲区
    frame_buffer = Stack(30 * 5)
    lock = threading.RLock()

    # rtsp_url = "rtsp://admin:quickhigh123456@192.168.3.155/h264/ch1/sub/av_stream"
    rtsp_url = 0
    t1 = threading.Thread(target=capture_thread, args=(rtsp_url, frame_buffer, lock))
    t1.start()
    t2 = threading.Thread(target=detect_thread, args=(cfg, frame_buffer, lock))
    t2.start()