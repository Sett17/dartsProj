import cv2 as cv
import math
import numpy as np

empty = cv.resize(cv.imread("0.png"), (858, 515))
dart1 = cv.resize(cv.imread("1.png"), (858, 515))
dart2 = cv.resize(cv.imread("2.png"), (858, 515))

cv.imshow('0', empty)

dart1 = cv.absdiff(empty, dart1)
cv.imshow('0', dart1)


cv.imshow('0', dart2)

cv.waitKey(0)