#!/usr/bin/env python

import numpy as np
from math import pi
import cv2
import cv
import sys

if len(sys.argv) > 1:
	fn = sys.argv[1]
	print 'loading %s ...' % fn
	img1 = cv2.imread(fn, 0)
	img = cv.LoadImage(fn, cv.CV_LOAD_IMAGE_GRAYSCALE)
	size = cv.GetSize(img)

	temp = cv.CreateImage(size, img.depth, img.nChannels)
	print temp
	cv.Smooth(img, temp)

	canny = cv2.Canny(temp, 50, 100)
	color_dst = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
	lines = cv2.HoughLinesP(canny, 1, pi/90, 20, np.array([]), 5)

	try:
		for line in lines[0]:
			cv2.line(color_dst, (line[0], line[1]), (line[2], line[3]), cv.RGB(255,0,0), 1, 8)
	except:
		pass

	print lines[0].size

	cv2.namedWindow("Original")
	cv2.imshow("Original", img)

	cv2.namedWindow('Lines image')
	cv2.imshow('Lines image', color_dst)

	cv2.waitKey()


else:
	print "Please give a image path"