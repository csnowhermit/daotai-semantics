import cv2

frame = cv2.imread("D:/testData/in/left.jpg")

frame1 = cv2.resize(frame, (1920, 1080))
print(frame1.shape)

frame2 = cv2.transpose(frame1, 270)
print(frame2.shape)

frame3 = frame2[::-1]

cv2.imwrite("D:/testData/in/out3.jpg", frame3)