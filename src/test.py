import cv2
import numpy as np

img = cv2.imread('imgs/szaszki.jpg')
img = cv2.resize(img,None,fx=0.4,fy=0.4)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# wykryte krawedzie
edges = cv2.Canny(gray,50,150)
#cv2.imwrite('edges.jpg',edges)

# pogrubienie
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)
#cv2.imwrite('dilation.jpg',dilation)

# rozmycie gaussa
gausiian = cv2.GaussianBlur(dilation,(5,5),0)


im2, contours, hierarchy = cv2.findContours(gausiian,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('gaussian.jpg',gausiian)


minLineLength = 100
maxLineGap = 50
lines = cv2.HoughLinesP(gausiian,1,np.pi/180,100,minLineLength,maxLineGap)

width, higth, channels = img.shape

blank_image = np.zeros((width, higth, channels), np.uint8)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),2)

#cv2.imshow('lines2.jpg',blank_image)

(thresh, im_bw) = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

