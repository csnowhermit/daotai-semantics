import sys
import time
import argparse
import cv2
import numpy as np
from yolo import YOLO
from PIL import Image
import timer
import face_recognition
from statistics import mode
from config import *
from utils.inference import *
from utils.preprocessor import preprocess_input

frame_interval = 3  # Number of frames after which to run face detection
fps_display_interval = 5  # seconds
frame_count = 0

emotion_window = []

print("face_detect:", face_detect)
portrait_log.logger.info("face_detect: %s" % (face_detect))

cap = cv2.VideoCapture(0)
start_time = time.time()
while True:
    ret, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame is None:
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

                x1, x2, y1, y2 = apply_offsets(box, emotion_offsets)  # 左右上下

                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), thickness=3)
                draw_text(box, frame, emotion_mode,
                          color, 0, -20, 1, 1)
        cv2.imshow("emotion_demo", frame)
        cv2.waitKey(1)

        # Check our current fps
        end_time = time.time()
        if (end_time - start_time) > fps_display_interval:
            frame_rate = int(frame_count / (end_time - start_time))
            start_time = time.time()
            frame_count = 0