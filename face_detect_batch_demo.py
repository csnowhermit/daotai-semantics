import os
import cv2
from mymodel import face_detect

'''
    批量人脸检测
'''


def face_detect_batch_test():
    imgpath = "D:/testData/local/"
    imgoutpath = "D:/testData/local_out/"

    for imgfile in os.listdir(imgpath):
        img_fullpath = os.path.join(imgpath, imgfile)
        img_out = imgoutpath + imgfile[0: -4] + "-out.jpg"

        print(img_fullpath)
        frame = cv2.imread(img_fullpath)

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

        cv2.imwrite(img_out, frame)

if __name__ == '__main__':
    face_detect_batch_test()
