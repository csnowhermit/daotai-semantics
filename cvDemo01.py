import cv2

#input_path = "rtsp://admin:quickhigh123456@192.168.0.103"
input_path = "rtsp://admin:quickhigh123456@192.168.120.155"

cap = cv2.VideoCapture(input_path)
print("cap:", cap)
if cap.isOpened():
    print("cap.isOpened()")
else:
    print("cap not opened")

while True:
    ret, frame = cap.read()
    if frame is None:
        print("frame is None")
    cv2.imshow("frame", frame)
    cv2.waitKey(1)