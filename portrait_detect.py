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
from utils.pyKinectUtil import Kinect

'''
    人物画像模块
'''




if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    FLAGS = parser.parse_args()

    yolo = YOLO(**vars(FLAGS))
    while True:
        try:
            # frame, facebboxes, landmarks = captureImage(input_webcam)  # frame <np.ndarray>
            frame, facebboxes, landmarks = captureImageFromKinect(Kinect())  # frame <np.ndarray>
            if frame is not None:
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