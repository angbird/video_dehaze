import socket
import cv2
import io
from PIL import Image
from Dark_Channel_Prior import get_dark_channel
from Dark_Channel_Prior import get_atmospheric_light
from Dark_Channel_Prior import get_transmittance_estimate
from Dark_Channel_Prior import transmittance_refine
from Dark_Channel_Prior import recover
import numpy as np


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
s.bind(("0.0.0.0", 9090))
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # 视频的编码
writer = cv2.VideoWriter("video_remove_haze.avi", fourcc, 5, (960, 320))  # 定义视频对象输出
while True:
    data, IP = s.recvfrom(100000)
    bytes_stream = io.BytesIO(data)
    image = Image.open(bytes_stream)
    img = np.asarray(image)
    unprocessed_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 传入待处理图片，ESP32采集的是RGB格式，要转换为BGR（opencv的格式）
    I = unprocessed_img.astype('float64') / 255
    dark = get_dark_channel(I, 15)
    A = get_atmospheric_light(I, dark)
    te = get_transmittance_estimate(I, A, 15)
    t = transmittance_refine(unprocessed_img, te)
    J = recover(I, t, A)
    contrast = np.hstack((I, J))
    cv2.imshow("contrast", contrast)
    frame_temp = (contrast*255).astype('uint8')   # 将待储存的图片转换为uint8格式
    # ret = writer.write(frame_temp)  # 存储视频
    # print(ret)
    # cv2.imshow("ESP32 Capture Image", img)
    if cv2.waitKey(1) == ord("q"):
        break
