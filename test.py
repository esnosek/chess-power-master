import cv2
import numpy as np

img = cv2.imread('szaszki.jpg')
#img = cv2.resize(img,None,fx=0.4,fy=0.4)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# wykryte krawedzie
edges = cv2.Canny(gray,50,150)
cv2.imwrite('edges.jpg',edges)

# pogrubienie
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)
cv2.imwrite('dilation.jpg',dilation)

# rozmycie gaussa
gausiian = cv2.GaussianBlur(dilation,(3,3),0)
cv2.imwrite('gaussian.jpg',gausiian)


minLineLength = 100
maxLineGap = 50
lines = cv2.HoughLinesP(gausiian,1,np.pi/180,100,minLineLength,maxLineGap)

width, higth, channels = img.shape

blank_image = np.zeros((width, higth, channels), np.uint8)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('lines.jpg',blank_image)
