import cv2 as cv
from cv2 import VideoCapture
from cv2 import imshow
from cv2 import resize
import random
from cv2 import Canny
from cv2 import dilate
from cv2 import contourArea
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
from pyparsing import restOfLine
import webcolors


cv.namedWindow("trackbar")


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def rgb_to_hex(r, g, b):
    hex = ('{:X}{:X}{:X}').format(r, g, b)
    return hex


def empty(x):
    pass

def getcontours(dilate):
     contours,_=cv.findContours(dilate,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
     for cnt in contours:
         
         contourArea=cv.contourArea(cnt)
         if contourArea>6000:
             cv.drawContours(result,contours,-1,(255,0,0),2)
             peri=cv.arcLength(cnt,True)
             approx=cv.approxPolyDP(cnt,0.02*peri,True)
             print(len(approx))
          
             (x,y,w,h)=cv.boundingRect(approx)
             
             if len(approx)==4:
                shape="Square"
                cv.putText(result,shape,(x+15,y-7),FONT_HERSHEY_COMPLEX,1,(255,155,0),2)
                cv.rectangle(result,(x,y),(x+w+20,y+h+20),(255,0,0),2)
             elif len(approx)==3:
                 shape="Triangle"
                 
                 
                 cv.putText(result,shape,(x+15,y-7),FONT_HERSHEY_COMPLEX,1,(255,155,0),2)
             cv.rectangle(result,(x,y),(x+w+20,y+h+20),(255,0,0),2)

             

cv.createTrackbar("hue min", "trackbar", 19, 255, empty)
cv.createTrackbar("hue max", "trackbar", 74, 255, empty)
cv.createTrackbar("sat min", "trackbar", 45, 255, empty)
cv.createTrackbar("sat max", "trackbar", 170, 255, empty)
cv.createTrackbar("val min", "trackbar", 161, 255, empty)
cv.createTrackbar("val max", "trackbar", 255, 255, empty)
cv.createTrackbar("in", "trackbar", 133, 43, empty)
cv.createTrackbar("ax", "trackbar", 200, 200, empty)

cam = VideoCapture(0)

while True:
    hsvmin = cv.getTrackbarPos("hue min", "trackbar")
    hsvmax = cv.getTrackbarPos("hue max", "trackbar")
    satmin = cv.getTrackbarPos("sat min", "trackbar")
    satmax = cv.getTrackbarPos("sat max", "trackbar")
    valmin = cv.getTrackbarPos("val min", "trackbar")
    valmax = cv.getTrackbarPos("val max", "trackbar")
    im = cv.getTrackbarPos("in", "trackbar")
    ax = cv.getTrackbarPos("ax", "trackbar")

    isTrue, video = cam.read()
    gray = cv.cvtColor(video, cv.COLOR_BGR2HSV)
   

    lower = np.array([hsvmin, satmin, valmin])
    upper = np.array([hsvmax, satmax, valmax])
    mask = cv.inRange(gray, lower, upper)
    # imshow("mask",mask)
    result = cv.bitwise_and(video, video, mask=mask)
    blur=cv.GaussianBlur(result,(7,7),1)
    grayy=cv.cvtColor(result,cv.COLOR_BGR2GRAY)
    Canny=cv.Canny(grayy,im,ax)
    kernel=np.ones((5,5))
    dilate=cv.dilate(Canny,kernel=kernel,iterations=1)
    getcontours(dilate)
    if isTrue==True:
        cv.resize(gray,(300,300))
        cv.imshow("canny",gray)
        show=stackImages(0.6,([result]))
        cv.imshow("main",result)
    
    

    # print(result.shape)
    # print(result[:])
    num = random.randint(0, 399)
    num1 = random.randint(0, 399)
    # r=(result[num,num1][0])
    # g=(result[num,num1][1])
    # b=(result[num,num1][2])
    val = list(result[num, num1])
    val0 = []
    if result[num, num1][0] == 0:
        print("no color detected")
    else:

        #  kal=kal.replace("(",",")
        #  print(kal)
        a = 0

        for h in val:
            # print(a)
            val0.append(val[a])
            a = a+1
            if a == 3:

                val0.reverse()
                val1 = tuple(val0)
                print(f"R,G,B={val1}")
                rgb=f"R,G,B={val1}"
                cv.imshow("sds",video)
                r, g, b = val1
                
                hexadecimal = (rgb_to_hex(r, g, b))
                hex = (f"Hexadecimal=#{hexadecimal}")
                print(hex)
            
            # cv.imshow("sd",dilate)    
            #    color_name=webcolors.rgb_to_name(kal1)

            #    rgb_per=(webcolors.rgb_to_rgb_percent(kal1))

    k = cv.waitKey(1)
    if k == ord('g'):
        break
