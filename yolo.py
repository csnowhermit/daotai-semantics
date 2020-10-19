# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
import cv2
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from utils.commonutil import *
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from mymodel import *
import traceback

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    '''
        检测图片
        :param image 被检图片，PIL.Image
        :param gray_image 被检图片的灰度图，np.ndarray
        :param bbox 人脸框
    '''
    def detect_image(self, image, gray_image, bboxes):
        start = timer()
        featureDict = {}    # 人物画像的项点和值

        try:
            if self.model_image_size != (None, None):
                assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
                assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
                boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            else:
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
                boxed_image = letterbox_image(image, new_image_size)
            image_data = np.array(boxed_image, dtype='float32')

            print(image_data.shape)
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
            draw = ImageDraw.Draw(image)

            personList = []  # 人列表
            luggageList = []  # 行李列表
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                if predicted_class == "person":
                    personList.append((predicted_class, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), score))
                elif predicted_class in personal_luggage_list:
                    luggageList.append((predicted_class, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), score))
                else:
                    continue

            # 绑定人脸和人体的关系
            askPersonList = bindFaceAndPerson(bboxes, personList)  # （人体框-上左下右，人体面积，人脸框-左上右下）

            # 对askPersonList进行过滤，找到具体的提问人
            if len(askPersonList) > 0:
                result = sorted(askPersonList, key=lambda x: x[1], reverse=True)  # reverse=True表示降序
                asker = result[0]  # 提问人
                print("asker:", asker)

                x0, x1, x2 = asker  # 三者分别是：人物框，人体框面积，人脸框
                person_top, person_left, person_bottom, person_right = x0  # 上左下右

                w = person_right - person_left
                h = person_bottom - person_top
                personNewBox = (person_top - expand_multiple * h, person_left - expand_multiple * w,
                                person_bottom + expand_multiple * h,
                                person_right + expand_multiple * w)  # 询问人的新框：行李检测有小框

                # 1.行李检测
                lugList = []
                for luggage in luggageList:  # 找行李，人物框扩大一倍，行李中心点在框内的，就算是携带的行李
                    p_class, box, score = luggage
                    lug_top, lug_left, lug_bottom, lug_right = box
                    w = lug_right - lug_left
                    h = lug_bottom - lug_top
                    luggageCenter = (lug_left + w / 2, lug_top + h / 2)  # 行李的中心点
                    tag = isin(luggageCenter, personNewBox)
                    if tag is True:
                        lugList.append((p_class, box, score))

                        # for i in range(3):    # 生产环境无需画框
                        #     draw.rectangle(
                        #         [lug_left + i, lug_top + i, lug_right - i, lug_bottom - i],
                        #         outline=(0, 0, 255))
                # print("personList: %s, luggageList: %s" % (personList, lugList))
                featureDict["luggage"] = lugList

                # 2.性别年龄检测
                face_left, face_top, face_right, face_bottom = x2  # 左上右下
                tmp = crop_face(image, x2, margin=40, size=face_size)  # 裁剪脑袋部分，并resize，image：<class 'PIL.Image.Image'>
                faces = [[face_left, face_top, face_right, face_bottom]]  # 做成需要的格式：[[], [], []]
                face_imgs = np.empty((len(faces), face_size, face_size, 3))
                # face_imgs[0, :, :, :] = cv2.resize(np.asarray(tmp), (face_size, face_size))    # PIL.Image转为np.ndarray，不resize会报错：ValueError: could not broadcast input array from shape (165,165,3) into shape (64,64,3)
                face_imgs[0, :, :, :] = tmp

                results = age_gender_model.predict(face_imgs)  # 性别年龄识别
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                featureDict["gender"] = "F" if predicted_genders[0][0] > 0.5 else "M"
                featureDict["age"] = int(predicted_ages[0])

                # 3.表情识别
                x1, x2, y1, y2 = apply_offsets(box, emotion_offsets)  # 左右上下
                print("emotion reco: ", type(gray_image), x1, x2, y1, y2)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    pass

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]

                featureDict["emotion"] = emotion_text  # 表情

                # 4.其他。。。后续待加
                # ====================

                print("portraitDict: ", featureDict)
        except Exception as e:
            traceback.print_exc()

        end = timer()
        print(end - start)
        return image, featureDict

    def close_session(self):
        self.sess.close()
