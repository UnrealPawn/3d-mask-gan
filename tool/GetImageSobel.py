#import pytesseract
from PIL import Image,ImageEnhance
import cv2
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
import numpy as np
def ImageToMatrix(image):
    matrix = np.asarray(image)
    return matrix


def matrixpic(img, x, y):
    matrixpic = ImageToMatrix(img)

    for i in range(500):
        for j in range(500):
            if matrixpic[i][j] > x:
                matrixpic[i][j] = 0
            elif matrixpic[i][j] < y:
                matrixpic[i][j] = 0
            else:
                matrixpic[i][j] = 255

    return matrixpic
def sobel(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst



def lines_rec(file,savepath):
    # 预处理
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
    zhongzhi = cv2.medianBlur(gray, 1)  # 中值滤波
    sobel1 = sobel(zhongzhi)  # sobel算子
    matrixpicd = matrixpic(sobel1, 255, 60)
    #return matrixpicd
    cv2.imwrite(savepath, matrixpicd)
    return matrixpicd
test_array = np.array([], dtype=float)
mat = lines_rec(r"C:\Users\Administrator\Desktop\untitled.png",r"C:\Users\Administrator\Desktop\untitled.png")
test_array = np.append(test_array,mat)
np.save('test_6.db', test_array)