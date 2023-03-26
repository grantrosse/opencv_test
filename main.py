import cv2 
import numpy as np 

##read image
image = cv2.imread('GreenScreen.jpg') 
##change to hsv
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
##mask for green
mask = cv2.inRange(hsv_image, (36, 25, 25), (70, 255,255))
## slice the green
imask = mask>0
green = np.zeros_like(image, np.uint8)
green[imask] = image[imask]
##change color scale
imgray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
##threshold to find contours
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
## get contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
## draw outline on the masked picture
output = cv2.drawContours(green, [max(contours, key = cv2.contourArea)], -1, (0,0,255), 20)
cv2.imwrite("final.png", output)