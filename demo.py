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

print(cv2.CAP_DSHOW)

box_areas = [100, 150, 140, 200, 120]
print(max(box_areas))
print(box_areas.index(max(box_areas)))

time_str = "1600312746347"
print(getFormatTime(time_str))