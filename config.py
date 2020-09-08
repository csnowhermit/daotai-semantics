import os
from Logger import *
import cv2

'''
    配置项
'''

# 导台ID
daotaiID = "center01"

# 日志文件
portrait_logfile = 'D:/data/portrait.log'
portrait_log = Logger(portrait_logfile, level='info')

# 随身物品
personal_luggage_list = ['bicycle', 'backpack',  'umbrella',
                         'handbag','tie', 'suitcase', 'skis', 'snowboard',
                         'sports ball', 'kite', 'baseball bat', 'baseball glove',
                         'skateboard', 'surfboard', 'tennis racket', 'bottle',
                         'laptop', 'cell phone', 'book', 'clock'
                        ]

# 摄像机地址
# input_webcam = "rtsp://admin:quickhigh123456@192.168.0.155/h264/ch1/sub/av_stream"    # 字码流接入
input_webcam = 0

# cap = cv2.VideoCapture(input_webcam)

# 现场图像保存路径
portrait_img_path = "D:/daotai/portrait_imgs/"

# 人脸面积最小阈值
face_area_threshold = 1000

# 询问人的有效框各扩大多少倍
expand_multiple=0.5

# 表情
emotion_offsets = (20, 40)

# 人脸大小
face_size = 64


# 表情标签
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

frame_window = 10
emotion_offsets = (20, 40)