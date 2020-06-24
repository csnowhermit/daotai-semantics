import cv2
import queue
import time
import threading

q = queue.Queue()

'''
    opencv多线程
    Receive：负责接入摄像头的视频帧
    Display：负责显示或处理
'''

def Receive():
    print("start Reveive")
    # cap = cv2.VideoCapture("rtsp://admin:quickhigh@192.168.120.155")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Display():
    print("Start Displaying")
    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    time.sleep(5)
    p2.start()
