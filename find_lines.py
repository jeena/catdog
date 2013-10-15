#!/usr/bin/env python

import cv2, cv, sys, math, os, numpy
from scipy.spatial import KDTree

def extractFeatures(label):

	directory = "img/" + label + "/"

	features = []

	for fn in os.listdir(directory):

		img = cv2.imread(directory + fn, 0)

		#temp = cv.CreateImage((100,100), cv.CV_8U, 1)
		#cv.Smooth(img, temp)

		canny = cv2.Canny(img, 50, 100)
		color_dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

		# find colored
		black_pixels = numpy.count_nonzero(img)

		# find lines lines
		lines = cv2.HoughLinesP(canny, 1, math.pi/360, 5, None, 10, 1)

		lengths = []
		angles = []
		try:
			for line in lines[0]:
				x1, y1, x2, y2 = line
				#cv2.line(color_dst, (x1, y1), (x2, y2), cv.RGB(255,0,0), 1, 8)
				length = int(math.sqrt(math.pow((x1-x2), 2) + math.pow((y1-y2), 2)))
				lengths.append(length)

				angle = int(math.degrees(math.atan((y1-y2) / (x1-x2))))
				angles.append(angle)
		except:
			pass

		# print out everything
		lines_count = len(lengths)
		mid_length = sum(lengths) / lines_count
		mid_angle = sum(angles) / lines_count

		features.append([[lines_count, mid_length, mid_angle, black_pixels], label])

		#cv2.namedWindow("Original")
		#cv2.imshow("Original", img)

		#cv2.namedWindow('Lines image ' + fn)
		#cv2.imshow('Lines image ' + fn, color_dst)

	return features


if __name__ == "__main__":
	cats = extractFeatures("cat")
	dogs = extractFeatures("dog")

	test_count = 5

	test_data = dogs[:test_count] + cats[:test_count] 
	test_labels = map(lambda a: a[1], test_data)
	test_features = map(lambda a: a[0], test_data)

	data = cats[test_count:] + dogs[test_count:]
	labels = map(lambda a: a[1], data)
	features = map(lambda a: a[0], data)

	tree = KDTree(features)
	
	for t in xrange(0, test_count * 2):
		d, i = tree.query(test_features[t], k=2)
		for j in xrange(0, len(i)):
			print test_labels[t] + " is predicted to be a " + labels[i[j]] + " j: " + str(i[j]) + " d: " + str(d[j])

