#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 02:33:16 2018

@author: nos
"""

from statistics import median
import cv2

img_path = 'imgs/gray.jpg'
gray = cv2.imread(img_path, 0)

gray_array = gray.reshape((gray.shape[0]*gray.shape[1]))
x = median(gray_array)

bw = cv2.threshold(gray, x, max(gray_array), cv2.THRESH_BINARY)
cv2.imwrite('tmp/median_bw.jpg', bw[1])

