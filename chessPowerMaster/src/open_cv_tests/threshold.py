#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:06:39 2018

@author: nos
"""
import cv2

def adaptive_thresh(img, block_size, c):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, c)
    
    
if __name__ == '__main__':
    img_path = 'imgs/szaszki.jpg'
    img = cv2.resize(cv2.imread(img_path), None, fx = 0.4, fy = 0.4)
    th_a = adaptive_thresh(img, 351, 2)
