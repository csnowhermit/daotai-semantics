import cv2
import numpy as np

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
    imgarray = np.asarray(image)  # PIL.Image 转 np.ndarray
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