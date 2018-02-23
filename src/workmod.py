#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:56:51 2018

@author: nos
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(list, weidth):
    bins=range(int(min(list)), int(max(list)) + weidth, weidth)
    plt.hist(list,bins)
    
def find_big_rectagles(contours):
    return [c for c in contours if cv2.contourArea(c) > 5000]

def find_areas(contours):
    return [cv2.contourArea(c) for c in contours]

def find_rectangles(contours):
    return [c for c in contours if is_rectangle(c)]

    
def is_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    rotaded_rect = cv2.boxPoints(rect)
    return cv2.contourArea(contour) / cv2.contourArea(rotaded_rect) > .95
    
def show_me_rectangles(cntrs):
    for i in range(1,len(cntrs)+1):
        blank = np.zeros(im2.shape, np.uint8)
        cv2.drawContours(blank, cntrs, i, (255), 1)
        cv2.imshow('c'.format(i), blank)
        cv2.waitKey()

"""
operation stream for rectange identification
"""

# step 1 read and convert to 
img_path = 'imgs/szaszki.jpg'
gray = cv2.resize(cv2.imread(img_path, 0), None, fx = 0.4, fy = 0.4)

cv2.imshow('23', gray)
# wykryte krawedzie, statyczne parametry !
edges = cv2.Canny(gray, 50, 150)

# pogrubienie
kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(edges, kernel,iterations = 1)

# rozmycie gaussa
gausiian = cv2.GaussianBlur(dilation, (5, 5), 0)

#kontury
im2, contours, hierarchy = cv2.findContours(gausiian, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('im2.jpg', im2)

#for i in range(1, len(contours)+1):
blank = np.zeros(im2.shape, np.uint8)
cv2.drawContours(blank, [box], -1, (255), 1)
cv2.imshow('c'.format(i), blank)

cont_stats = [(c, is_rectangle(c)) for c in contours]
cont_stats.sort(key = lambda x: x[1], reverse = True)
cntrs = [c[0] for c in cont_stats]

for i in range(1,len(cntrs)+1):
    blank = np.zeros(im2.shape, np.uint8)
    cv2.drawContours(blank, cntrs, i, (255), 1)
    cv2.imshow('c'.format(i), blank)
    cv2.waitKey()
