from Dark_Channel_Prior import get_dark_channel
from Dark_Channel_Prior import get_transmittance_estimate
from Dark_Channel_Prior import get_atmospheric_light
from Dark_Channel_Prior import transmittance_refine
from Dark_Channel_Prior import recover
import cv2
import numpy as np


def video_remove_haze(video_address):
    cap = cv2.VideoCapture(video_address)  # 生成视频读取对象
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频的编码
    writer = cv2.VideoWriter("video_remove_haze.avi", fourcc, fps, (width, height))  # 定义视频对象输出
    while cap.isOpened():
        ret, frame = cap.read()  # 读取视频画面
        # 对帧图像进行处理
        I = frame.astype('float32') / 255
        dark = get_dark_channel(I, 15)
        A = get_atmospheric_light(I, dark)
        te = get_transmittance_estimate(I, A, 15)
        t = transmittance_refine(frame, te)
        J = recover(I, t, A)
        contrast = np.hstack((I, J))
        cv2.imshow("contrast", contrast)   # 显示画面
        key = cv2.waitKey(1)
        # writer.write(J*255)  # 视频保存
        # 按Q退出
        if key == ord('q'):
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 释放所有显示图像窗口


if __name__ == '__main__':
    video_remove_haze(0)
