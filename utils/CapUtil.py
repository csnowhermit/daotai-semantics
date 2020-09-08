import os
import cv2
import threading

'''
    自定义栈，解决视频消费速度赶不上生成速度的问题。
    原因：opencv自带的缓冲区无提供api供操作
'''
class Stack:
    def __init__(self, stack_size):
        self.items = []
        self.stack_size = stack_size

    def is_empty(self):
        return len(self.items) == 0

    def pop(self):
        return self.items.pop()

    def peek(self):
        if not self.isEmpty():
            return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def push(self, item):
        if self.size() >= self.stack_size:
            # for i in range(0, self.size()):
            #     self.items.remove(self.items[0])
            self.items.clear()
        self.items.append(item)

    # 清空缓冲区
    def clear(self):
        self.items.clear()

def capture_thread(input_webcam, frame_buffer, lock):
    if input_webcam == "0":
        input_webcam = int(0)
    print("capture_thread start")
    vid = cv2.VideoCapture(input_webcam)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while True:
        return_value, frame = vid.read()
        if return_value is not True:
            break
        lock.acquire()
        frame_buffer.push(frame)
        lock.release()
        cv2.waitKey(25)

def show_thread(frame_buffer, lock):
    print("detect_thread start")
    print("detect_thread frame_buffer size is", frame_buffer.size())

    while True:
        if frame_buffer.size() > 0:
            lock.acquire()
            frame = frame_buffer.pop()
            lock.release()
            # 每次拿最新的，显示
            cv2.imshow("result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    frame_buffer = Stack(30 * 5)
    lock = threading.RLock()
    t1 = threading.Thread(target=capture_thread, args=(0, frame_buffer, lock))
    t1.start()
    t2 = threading.Thread(target=show_thread, args=(frame_buffer, lock))
    t2.start()