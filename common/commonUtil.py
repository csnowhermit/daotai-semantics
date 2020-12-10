import numpy as np

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
    :return askPersonList[]，tuple(人物框，人物面积，人脸框，人脸面积)
'''
def bindFaceAndPerson(bboxes, personList):
    askPersonList = []    # 可能的提问者列表

    used_faceList = []    # 已使用过的脸框
    used_personList = []    # 已使用过的人体框
    for face_box in bboxes:
        face_box_line = "%s_%s_%s_%s" % (str(face_box[0]), str(face_box[1]), str(face_box[2]), str(face_box[3]))
        if face_box_line in used_faceList:    # 已经比较过的无须再参与比较
            continue
        left, top, right, bottom = face_box
        w = right - left
        h = bottom - top
        faceCenter = (left + w / 2, top + h / 2)    # 脸的中心点
        face_area = w * h

        for i in range(len(personList)):
            p_class, person_id, person_box, score, track_state = personList[i]    # 类别，id，人物框，得分
            person_box_line = "%s_%s_%s_%s" % (str(person_box[0]), str(person_box[1]), str(person_box[2]), str(person_box[3]))
            if person_box_line in used_personList:
                continue

            top, left, bottom, right = person_box
            person_area = (right - left) * (bottom - top)    # w*h

            tag = isin(faceCenter, person_box)
            if tag is True:
                askPersonList.append((p_class, person_id, person_box, person_area, face_box, face_area, score, track_state))

                used_faceList.append(face_box_line)
                used_personList.append(person_box_line)
                break
    return askPersonList


'''
    计算iou
'''
def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union