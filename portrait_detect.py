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

'''
    人物画像模块
'''


'''
    抓取图片
'''
def captureImage(input_webcam):
    if input_webcam == "0":
        input_webcam = int(0)

    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_count = 0

    print("face_detect:", face_detect)
    portrait_log.logger.info("face_detect: %s" % (face_detect))

    cap = cv2.VideoCapture(input_webcam)
    start_time = time.time()
    retry = 0    # 读cap.read()重试次数
    while True:
        ret, frame = cap.read()

        if frame is None:
            retry += 1
            time.sleep(0.5)    # 读取失败后立马重试没有任何意义
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    FLAGS = parser.parse_args()

    yolo = YOLO(**vars(FLAGS))
    while True:
        try:
            frame, facebboxes, landmarks = captureImage(input_webcam)  # frame <np.ndarray>
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("frame:", type(frame), frame.shape)
            portrait_log.logger.info(
                "********** start a portrait detect ********** %s" % (getFormatTime(str(int(time.time())))))
            detect_portrait(yolo=yolo, frame=frame, gray_image=gray_image, bbox=facebboxes,
                            landmarks=landmarks)  # 应用中调这行代码即可
            portrait_log.logger.info(
                "========== end a portrait detect ========== %s" % (getFormatTime(str(int(time.time())))))
        except Exception as e:
            portrait_log.logger.error(traceback.format_exc())