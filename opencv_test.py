# -*-coding:utf-8-*-
import os
import numpy as np
import cv2


root_path = '/media/susu/588A39568A3931BC/Data'


def get_camera():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 编码格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 写文件
    out = cv2.VideoWriter(os.path.join(root_path, 'output.avi'),fourcc, 20.0, (640,480))

    while (True):
        # 读取摄像头
        ret, frame = cap.read()
        # 灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def write_mask():
    img = cv2.imread(os.path.join(root_path, 'watch.jpg'), cv2.IMREAD_COLOR)
    # 画线条
    cv2.line(img, (0, 0), (150, 150), (255, 255, 255), 15)
    # 画矩形
    cv2.rectangle(img, (15, 25), (200, 150), (0, 0, 255), 15)
    # 画圆
    cv2.circle(img, (100, 63), 55, (0, 255, 0), -1)
    # 画多边形
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], True, (0, 255, 255), 3)
    # 写字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV Tuts!', (0, 130), font, 1, (200, 255, 155), 2, cv2.LINE_AA)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def modify_img():
    img = cv2.imread(os.path.join(root_path, 'watch.jpg'), cv2.IMREAD_COLOR)
    # print(img.shape)
    # print(img.size)
    # print(img.dtype)
    # px = img[55,55]
    # print(px)
    # img[55,55] = [255, 255, 255]
    # img[100:150, 100:150] = [255, 255, 255]
    watch_face = img[37:111, 107:194]
    img[0:74, 0:87] = watch_face
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def operation_img():
    img1 = cv2.imread(os.path.join(root_path, '3D-Matplotlib.png'))
    img2 = cv2.imread(os.path.join(root_path, 'mainsvmimage.png'))
    img3 = cv2.imread(os.path.join(root_path, 'mainlogo.png'))

    # add = img1 + img2
    # cv2直接add会造成想数值过大
    # add = cv2.add(img1, img2)
    # 权重相加
    # add = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
    # cv2.imshow('add', add)

    # 创建尺寸相同的ROI
    rows, cols, channels = img3.shape
    roi = img1[0:rows, 0:cols]

    img3gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    # 设置阈值
    ret, mask = cv2.threshold(img3gray, 220, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    img3_fg = cv2.bitwise_and(img3, img3, mask=mask)

    dst = cv2.add(img1_bg, img3_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('res', img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def use_threshold():
    img = cv2.imread(os.path.join(root_path, 'bookpage.jpg'))
    #二元阈值， 阈值为 125（最大为 255），那么 125 以下的所有内容都将被转换为 0 或黑色，而高于 125 的所有内容都将被转换为 255 或白色
    retval1, threshold1 = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
    # 先转灰度，在使用二元阈值
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval2, threshold2 = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
    # 自适应阈值
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # 大津阈值
    retval3, threshold3 = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('original', img)
    cv2.imshow('threshold1', threshold1)
    cv2.imshow('threshold2', threshold2)
    cv2.imshow('Adaptive threshold', th)
    cv2.imshow('Otsu threshold', threshold3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    use_threshold()
# https://www.jianshu.com/p/91f5bce4420d


