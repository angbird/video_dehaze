import cv2
import math
import numpy as np


def get_dark_channel(image, size):     # 得到暗通道图像
    image_b, image_g, image_r = cv2.split(image)
    temp_image = cv2.min(image_b, cv2.min(image_g, image_r))  # 在图像的各个像素位置上取得rgb分量的最小值，返回为灰度图
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel_image = cv2.erode(temp_image, kernel)
    return dark_channel_image


def get_atmospheric_light(image, dark_channel_image):  # 求大气光值
    [height, weight] = image.shape[:2]     # 获取图片宽度和高度
    image_size = height*weight                    # 获取图像大小
    needed_pixels_num = int(max(math.floor(image_size/1000), 1))  # 获取图像0.1%的像素个数,设置下限为1
    dark_channel_vector = dark_channel_image.reshape(image_size, 1)
    dark_channel_vector = [dark_channel_vector[i][0] for i in range(0, image_size)]  # 将暗通道图转为一维向量
    image_2_dimension = image.reshape(image_size, 3)      # 将原图转为二维矩阵，rgb三通道
    index = np.array(dark_channel_vector).argsort()                     # 返回暗通道向量从小到大的索引值
    index = index[image_size-needed_pixels_num::]     # 返回图像最亮的0.1%像素点索引值
    atm_light_sum = np.zeros([1, 3])                         # 创建零矩阵存放大气光值
    for i in range(0, needed_pixels_num):           # 在原图中找到最亮的0.1%个像素点，相加求平均值，得到大气光值
        atm_light_sum = atm_light_sum+image_2_dimension[index[i]]
    atmospheric_light = atm_light_sum / needed_pixels_num    # 得到大气光值
    return atmospheric_light


def get_transmittance_estimate(image, atmospheric_light, size):  # 得到透射率估值
    omega = 0.95
    im3 = np.empty(image.shape, image.dtype)
    for i in range(0, 3):
        im3[:, :, i] = image[:, :, i]/atmospheric_light[0, i]
    transmittance_estimate = 1-omega*get_dark_channel(im3, size)
    return transmittance_estimate


def guided_filter(im, p, r, eps):     # 导向滤波
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def transmittance_refine(image, transmittance_estimate):   # 透射率完善
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    transmittance = guided_filter(gray, transmittance_estimate, r, eps)
    return transmittance


def recover(image, transmittance, atmospheric_light, tx=0.1):  # 返回去雾图片
    result = np.empty(image.shape, image.dtype)           # 创建与原图大小类型相同的空矩阵
    t = cv2.max(transmittance, tx)                                  # 透射率下限为0.1
    for i in range(0, 3):
        result[:, :, i] = (image[:, :, i] - atmospheric_light[0, i]) / t + atmospheric_light[0, i]
    return result


if __name__ == '__main__':
    haze_image = cv2.imread('fj-wu.png')
    I = haze_image.astype('float64') / 255
    dark = get_dark_channel(I, 15)
    A = get_atmospheric_light(I, dark)
    te = get_transmittance_estimate(I, A, 15)
    t = transmittance_refine(haze_image, te)
    J = recover(I, t, A)
    contrast = np.hstack((I, J))
    cv2.imshow("contrast", contrast)
    cv2.waitKey(0)
    cv2.imwrite('haze_and_dehaze.jpg', contrast*255)
