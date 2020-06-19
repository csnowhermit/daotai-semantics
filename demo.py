import time
import datetime
from PIL import Image
from utils.commonutil import getFormatTime
import json

# print(getFormatTime(time.time()))

with open("../", encoding='utf-8') as fo:
    for i, line in enumerate(fo.readlines()):
        print(line.split(",")[0])