import numpy as np
from PIL import Image
import os
import cv2


out_path =r"E:\dataset\new yixian\rgb\cancer"
def cvtcolor(image):
    try:
        if len(np.shape(image))==3 and np.shape(image)[2]==3:
            return image
        else:
            image=image.convert('RGB')
            return image
    except:
        print(filename)

for filename in os.listdir(out_path):                          # 遍历输入路径，得到图片名
    print(filename)
    cvtcolor(filename)


