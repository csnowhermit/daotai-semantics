import os
import cv2
import time
from mymodel import face_detect

from utils.CapUtil import Stack
from utils.dateUtil import formatTimestamp
import threading

'''
    人脸检测：亚克力板后摄像头人脸检测
'''

frame_buffer = Stack(30 * 5)
lock = threading.RLock()

def Receive():
    # print("start Receive")
    # print("start Receive")
    rtsp_url = "rtsp://admin:quickhigh123456@192.168.0.201/h264/ch1/sub/av_stream"
    # rtsp_url = 0
    cap = cv2.VideoCapture(rtsp_url)
    # cap = getCap(input_webcam)
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
    print("face_detect:", face_detect)

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
                for i in range(0, len(bboxes)):
                    box = bboxes[i]
                    left, top, right, bottom = box
                    w = right - left
                    h = bottom - top
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)

            cv2.imshow("frame", frame)
            cv2.waitKey(1)

            savefile = "D:/testData/" + formatTimestamp(time.time(), format="%Y%m%d_%H%M%S", ms=True) + ".jpg"
            cv2.imwrite(savefile, frame)

if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=percept)
    p1.start()
    time.sleep(5)
    p2.start()
