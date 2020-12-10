import os
from common.config import personal_luggage_list

'''
    对原始检出框及类型进行清洗
    :param bbox_xyxy 原始检出框：左上右下
    :param cls_conf 每个框的置信度
    :param cls_ids 每个框的类别序号
    :param class_names 所有类别集合：通过类别序号从集合中拿到类别名称
    :return 大人、小孩、物品分别的类别、框、置信度，其中，大人和小孩为：左上宽高；物品为：上左下右
'''
def cleaning_box(bbox_xyxy, cls_conf, cls_ids, class_names):
    person_classes = []  # 人
    person_boxs = []
    person_scores = []

    other_classes = []  # 物品
    other_boxs = []
    other_scores = []

    # 1.按需做框格式的转换
    for (xyxy, score, id) in zip(bbox_xyxy, cls_conf, cls_ids):
        predicted_class = class_names[id]    # 类别
        # print("原始检出：%s %s %s" % (predicted_class, xyxy, score))
        # log.logger.info("原始检出：%s %s %s" % (predicted_class, xyxy, score))

        if predicted_class == "person":  # 如果是人，只有在有效区域内才算
            person_classes.append(predicted_class)
            left, top, right, bottom = xyxy
            person_boxs.append([left, top, right - left, bottom - top])  # 左上宽高
            person_scores.append(score)
        elif predicted_class in personal_luggage_list:  # 其他类别的格式：上左下右
            other_classes.append(predicted_class)
            left, top, right, bottom = xyxy
            other_boxs.append([top, left, bottom, right])
            other_scores.append(score)
        else:
            pass
    return (person_classes, person_boxs, person_scores), (other_classes, other_boxs, other_scores)

