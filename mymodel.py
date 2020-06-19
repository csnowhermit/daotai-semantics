import os
import face_recognition
from keras.utils.data_utils import get_file
from keras.models import load_model
from wide_resnet import WideResNet
from config import face_size


# 人脸检测
face_detect = face_recognition.FaceDetection()  # 初始化mtcnn

# 性别年龄识别模型
WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"
age_gender_model = WideResNet(face_size, depth=16, k=8)()
age_gender_model_dir = os.path.join(os.getcwd(), "model_data").replace("//", "\\")
fpath = get_file('weights.18-4.06.hdf5',
                 WRN_WEIGHTS_PATH,
                 cache_subdir=age_gender_model_dir)
age_gender_model.load_weights(fpath)

# 表情识别
emotion_model_path = "./model_data/fer2013_mini_XCEPTION.102-0.66.hdf5"

emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]