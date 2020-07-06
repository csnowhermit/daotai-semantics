import time
import datetime
from PIL import Image
from utils.commonutil import getFormatTime
import json
import cv2

# print(getFormatTime(time.time()))

comming = {}

properties = {}

properties["gender"] = "M"
properties["age"] = 21

comming["sentences"] = properties
print(comming)
print(json.loads(str(comming), encoding='utf-8'))


print(cv2.CAP_DSHOW)