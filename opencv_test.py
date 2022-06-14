# -*-coding:utf-8-*-
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


# root_path = '/media/susu/588A39568A3931BC/Data'
root_path = './data'


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


def color_filter():
    img = cv2.imread(os.path.join(root_path, 'red_cap.JPG'), cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])
    # HSV，色调饱和度纯度
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fuzzy_smooth():
    img = cv2.imread(os.path.join(root_path, 'red_cap.JPG'), cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])
    # HSV，色调饱和度纯度
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)
    # 计算像素块均值
    kernel = np.ones((15, 15), np.float32) / 225
    smoothed = cv2.filter2D(res, -1, kernel)
    # 高斯模糊
    blur = cv2.GaussianBlur(res,(15,15),0)
    # 中值模糊
    median = cv2.medianBlur(res,15)
    # 双向模糊
    bilateral = cv2.bilateralFilter(res,15,75,75)

    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('Averaging', smoothed)
    cv2.imshow('Gaussian Blurring', blur)
    cv2.imshow('Median Blur',median)
    cv2.imshow('bilateral Blur',bilateral)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_transform():
    img = cv2.imread(os.path.join(root_path, 'red_cap.JPG'), cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)
    # 腐蚀，使用滑块（核）滑动，如果所有的像素是白色的，那么我们得到白色，否则是黑色。 这可能有助于消除一些白色噪音
    erosion = cv2.erode(mask, kernel, iterations=1)
    # 膨胀，让滑块滑动，如果整个区域不是黑色的，就会转换成白色
    dilation = cv2.dilate(mask, kernel, iterations=1)
    # 开放，消除“假阳性”
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 关闭，消除假阴性
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('img', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Erosion', erosion)
    cv2.imshow('Dilation', dilation)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gradual_edge():
    # 渐变和边缘检测
    img = cv2.imread(os.path.join(root_path, 'red_cap.JPG'), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.Canny(img,100,200)

    cv2.imshow('img', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    cv2.imshow('Edges',edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def match_template():
    img_rgb = cv2.imread(os.path.join(root_path, 'opencv-template-matching-python-tutorial.jpg'))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread(os.path.join(root_path, 'opencv-template-for-matching.jpg'), 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # 设置匹配阈值
    threshold = 0.8
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    cv2.imshow('Detected', img_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grabcut():
    # 前景提取
    img = cv2.imread(os.path.join(root_path, 'opencv-python-foreground-extraction-tutorial.jpg'))
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (161, 79, 150, 150)

    # 像素 0 和 2 转换为背景，而像素 1 和 3 现在是前景
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    plt.imshow(img)
    plt.colorbar()
    plt.show()


def angle_detection():
    # 角点检测
    img = cv2.imread(os.path.join(root_path, 'opencv-corner-detection-sample.jpg'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    cv2.imshow('Corner', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def feature_match():
    # 特征匹配(单映射)，爆破
    img1 = cv2.imread(os.path.join(root_path, 'opencv-feature-matching-template.jpg'), 0)
    img2 = cv2.imread(os.path.join(root_path, 'opencv-feature-matching-image.jpg'), 0)

    # ORB探测器
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # 匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # 绘制前十个匹配结果
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.imshow(img3)
    plt.show()


def mobile_detection():
    # 从静态背景中提取移动的前景,也可以用来比较两个相似的图像，并提取差异
    cap = cv2.VideoCapture(os.path.join(root_path, 'people-walking.mp4'))
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        cv2.imshow('fgmask', frame)
        cv2.imshow('frame', fgmask)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def cascade_detection():
    # 检测文件地址 https://github.com/Itseez/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier(os.path.join(root_path, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(root_path, 'haarcascade_eye.xml'))

    img = cv2.imread(os.path.join(root_path, 'peoples.JPG'), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cascade_detection()
# https://www.jianshu.com/p/91f5bce4420d


