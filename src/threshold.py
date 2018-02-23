#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:06:39 2018

@author: nos
"""
import cv2
import numpy as np

def adaptive_thresh(img_gray, block_size, c):
    
    return cv2.adaptiveThreshold(img_gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, c)
    
    
if __name__ == '__main__':
    path = 'imgs/szaszki.jpg'
    gray = cv2.imread(path, 0)
    gray = cv2.resize(gray, (0,0), fx=.4, fy=.4)
    th_a = adaptive_thresh(gray, 351, 2)
    
    im2, contours, hierarchy = cv2.findContours(th_a,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    width, higth = th_a.shape
    blank_image = np.zeros((width, higth), np.uint8)
    cv2.drawContours(blank_image, contours, 3, (0,255,0), 7)
    
    cv2.imshow('2', th_a)