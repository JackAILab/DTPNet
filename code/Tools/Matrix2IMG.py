import numpy as np
import cv2

# input_train = np.random.randint(low=0, high=255, size=(32, 32, 1), dtype=np.uint8)
# input_train = input
# input_train = x
input_train = a[:,3:6].detach()
# input_train = x_final[:,3:6].detach()
# input_train = x_real[:,0:3].detach()
# == 0. Original IMG
'''
图像处理中有多种色彩空间，例如 RGB、HLS、HSV、HSB、YCrCb、CIE XYZ、CIE Lab 等，经常要遇到色彩空间的转化，以便生成 mask 图等操作.
对于亮度敏感的为: RGB2HSV  RGB2LAB  RGB2HSL

对比度：指不同颜色之间的差别。对比度=最大灰度值/最小灰度值
亮度：这个容易理解，就是让图像色彩更加鲜亮
锐度：即清晰度，它是反映图像平面清晰度和图像边缘锐利程度的一个指标
色度：色彩的纯度,也叫饱和度或彩度
'''
data = input_train*255
data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
data = np.array(data, dtype='uint8')

cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/Original_add.png', data)

# # == 1. COLOR_RGB2BGR # dataset 准备的时候 已经进行了BGR2RGB的转换
# data = input_train*255
# data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
# data = np.array(data, dtype='uint8')

# data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

# cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/COLOR_RGB2BGR.png', data)


# # == 2. COLOR_RGB2HSV
# '''
# HSI 颜色空间可以用一个圆锥空间模型来描述，能清晰表现色调(Hue)、饱和度(Saturation, Chroma)和亮度(Intensity, Brightness)的变化情形.
# 色相 H(Hue) - 表示颜色的相位角. 红、绿、蓝分别相隔 120 度；互补色分别相差 180 度，即颜色的类别.
# 饱和度 S(Saturation) - 色彩的强度或纯度. 表示成所选颜色的纯度和该颜色最大的纯度之间的比率，范围：[0, 1]，即颜色的深浅程度.
# 亮度 I(Intensity) - 表示颜色的明亮程度，通常以 0% (黑色) 到 100% (白色) 的百分比来衡量(人眼对亮度很敏感).
# HSI色彩空间和RGB色彩空间只是同一物理量的不同表示法，它们之间可以进行相互转换：
# HSI颜色模式中的色调使用颜色类别表示，饱和度与颜色的白光光亮亮度刚好成反比，代表灰色与色调的比例，亮度是颜色的相对明暗程度.
# '''
# data = input_train*255
# data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
# data = np.array(data, dtype='uint8')

# data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)

# cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/COLOR_RGB2HSVtett.png', data)


# # == 3. COLOR_RGB2GRAY
# '''
# 为什么图像特征提取的时候，要用灰度图像呢？最直接的原因：减少计算量
# '''
# data = input_train*255
# data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
# data = np.array(data, dtype='uint8')

# data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

# cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/COLOR_RGB2GRAY.png', data)


# # == 4. COLOR_RGB2YCrCb
# '''
# YCrCb也称为YUV，主要用于优化彩色视频信号的传输。与RGB视频信号传输相比，它最大的优点在于只需占用极少的频宽（RGB要求三个独立的视频信号同时传输）。
# 其中“Y”表示明亮度（Luminance或Luma），也就是灰阶值；
# 而“U”和“V” 表示的则是色度（Chrominance或Chroma），作用是描述影像色彩及饱和度，用于指定像素的颜色。
# 其中，Cr反映了RGB输入信号红色部分与RGB信号亮度值之间的差异。而Cb反映的是RGB输入信号蓝色部分与RGB信号亮度值之间的差异。
# YCrCb颜色空间在做肤色检测上有很好的效果，比HSV颜色空间要好一些
# 查阅一些网上的资料得知，正常黄种人的Cr分量大约在140 ~ 175之间，Cb分量大约在100 ~ 120之间。
# '''
# data = input_train*255
# data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
# data = np.array(data, dtype='uint8')

# data = cv2.cvtColor(data, cv2.COLOR_RGB2YCrCb)

# cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/COLOR_RGB2YCrCb.png', data)


# # == 5. COLOR_RGB2HLS
# '''
# HSL 模式和 HSV(HSB) 都是基于 RGB 的，是作为一个更方便友好的方法创建出来的。
# HSB 为 色相，饱和度，明度；
# HSL 为 色相，饱和度，亮度，
# HSV 为 色相，饱和度，明度。
# '''
# data = input_train*255
# data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
# data = np.array(data, dtype='uint8')

# data = cv2.cvtColor(data, cv2.COLOR_RGB2HLS)

# cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/COLOR_RGB2HLS.png', data)


# # == 6. COLOR_RGB2XYZ
# '''
# XYZ 色彩空间是为了解决更精确地定义色彩而提出来的， XYZ 三个分量中， XY代表的是色度，
# 其中Y分量既可以代表亮度也可以代表色度， 三个分量的单位都是 cm/m2 ， （或者叫做nit）。
# 我们无法用RGB来精确定义颜色， 因为，不同的设备显示的RGB都是不一样的，不同的设备， 显示同一个RGB， 在人眼看出来是千差万别的，
# 如果我们用XYZ定义一个设备的色彩空间， 这样就精确多了！
# '''
# data = input_train*255
# data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
# data = np.array(data, dtype='uint8')

# data = cv2.cvtColor(data, cv2.COLOR_RGB2XYZ)

# cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/COLOR_RGB2XYZ.png', data)


# # == 7. COLOR_RGB2LAB
# '''
# LAB 相较于RGB与CMYK等颜色空间更符合人类视觉，也更容易调整：想要调节亮度，就调节L 通道，想要调节只色彩平衡就分别调 A 和 B.
# 理论上说，L、A、B 都是实数，一般限定在一个整数范围内：L 越大，亮度越高。L 为 0 时代表黑色，为100时代表白色。
# A 从负数变到正数，对应颜色从绿色变到红色。B 从负数变到正数，对应颜色从蓝色变到黄色。
# 在实际应用中常常将颜色通道的范围[-100, +100]或[-128, 127]之间。
# '''
# data = input_train*255
# data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
# data = np.array(data, dtype='uint8')

# data = cv2.cvtColor(data, cv2.COLOR_RGB2LAB)

# cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/COLOR_RGB2LAB.png', data)


# # == 8. COLOR_RGB2YUV
# '''
# YUV是一种色彩编码模式，其中Y表示亮度（Luminance），也就是灰度值，UV分别表示色度（Chrominance）和浓度（Chroma），作用是描述图像色彩和饱和度，用于指定像素的颜色。

# YUV设计初衷是为了解决彩色电视机与黑白电视的兼容性，从rgb的颜色空间，转换为yuv的颜色空间。
# 其利用了人类眼睛的生理特性（人眼对亮度变化的敏感性高于对颜色变化的敏感性），允许降低色度的带宽，降低了传输带宽。
# '''
# data = input_train*255
# data = data[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
# data = np.array(data, dtype='uint8')

# data = cv2.cvtColor(data, cv2.COLOR_RGB2YUV)

# cv2.imwrite('/data/ProjectData/Derain/Rain200L/TrainedModel/mixTrans_Pro_LSTM/Logs/Results/COLOR_RGB2YUV.png', data)





