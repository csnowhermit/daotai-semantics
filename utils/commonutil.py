import os
import cv2
import datetime
import numpy as np

'''
    通用工具
'''

'''
    判断某点是否在某框内
    :param luggageCenter 行李框中心点
    :param personBox 行人框边界，上左下右 
    :return 返回True，说明行李在指定人框内，可认为是这个人的行李
'''
def isin(center, personBox):
    top, left, bottom, right = personBox
    centerx, centery = center

    if (centerx >= left and centerx <= right) and (centery >= top and centery <= bottom):
        return True
    else:
        return False

'''
    绑定人脸和人体的关系
    :param bboxes 人脸框（左上右下）
    :param personList 人体列表
    :return askPersonList[]，tuple(人物框，人物面积，人脸框)
'''
def bindFaceAndPerson(bboxes, personList):
    askPersonList = []    # 可能的提问者列表

    used_faceList = []    # 已使用过的脸框
    used_personList = []    # 已使用过的人体框
    for face in bboxes:
        if face in used_faceList:    # 已经比较过的无须再参与比较
            continue
        left, top, right, bottom = face
        w = right - left
        h = bottom - top
        faceCenter = (left + w / 2, top + h / 2)    # 脸的中心点

        for person in personList:
            p_class, person_box, score = person
            if person_box in used_personList:
                continue

            top, left, bottom, right = person_box
            person_area = (right - left) * (bottom - top)    # w*h

            tag = isin(faceCenter, person_box)
            if tag is True:
                askPersonList.append((person_box, person_area, face))

                used_faceList.append(face)
                used_personList.append(person_box)
                break
    return askPersonList

'''
    裁剪人脸并resize
    :param image 原图，<class 'PIL.Image.Image'>
    :param facebox 脸框
    
'''
def crop_face(image, facebox, margin=40, size=64):
    """
    :param imgarray: full image
    :param section: face detected area (left, top, right, bottom)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    imgarray = np.asarray(image)    # PIL.Image 转 np.ndarray
    img_h, img_w, _ = imgarray.shape
    if facebox is None:
        section = [0, 0, img_w, img_h]
    # (x, y, w, h) = section
    left, top, right, bottom = facebox
    margin = int(min(right - left, bottom - top) * margin / 100)
    x_a = left - margin
    y_a = top - margin
    x_b = right + margin
    y_b = bottom + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w - 1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h - 1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img


'''
    时间戳转格式化后时间
'''
def getFormatTime(timestamp):
    timestamp = str(timestamp)
    if len(timestamp) == 13:    # 分别处理13位和10位时间戳
        now = int(timestamp)
        d = datetime.datetime.fromtimestamp(now/1000)
    else:
        now = int(timestamp)
        d = datetime.datetime.fromtimestamp(now)
    formatTimeStr = d.strftime("%Y%m%d%H%M%S%f")
    return formatTimeStr
