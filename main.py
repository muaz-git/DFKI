from scipy import weave
import numpy as np
import cv2
import sys

IMAGE_HEIGHT = 998.0

def _thinningIteration(im, iter):
    I, M = im, np.zeros(im.shape, np.bool)
    expr = """
	for (int i = 1; i < NI[0]-1; i++) {
		for (int j = 1; j < NI[1]-1; j++) {
			int p2 = I2(i-1, j);
			int p3 = I2(i-1, j+1);
			int p4 = I2(i, j+1);
			int p5 = I2(i+1, j+1);
			int p6 = I2(i+1, j);
			int p7 = I2(i+1, j-1);
			int p8 = I2(i, j-1);
			int p9 = I2(i-1, j-1);

			int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
			         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
			         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
			         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
				M2(i,j) = 1;
			}
		}
	}
	"""
    weave.inline(expr, ["I", "iter", "M"])
    # print 'Type of I = ', type(I)
    # print 'Type of M = ', type(M)
    # print np.amin(I)
    # print np.amax(I)
    # print I.dtype
    # print M.dtype
    # return (np.float64(I) & np.float64(~M))
    return (I & ~M)
    # cv2.imshow('~M', M)
    # cv2.imshow('thinned', thinned)
    # cv2.imshow('original', v_original)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return cv2.bitwise_and(np.float64(I), np.float64(~M))

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def thinning(src):
    dst = src.copy()
    dst = im2double(dst)
    print src.dtype
    print dst.dtype
    # print "min ", np.amin(dst)
    # print "max ", np.amax(dst)
    prev = np.zeros(src.shape[:2], np.uint8)
    diff = None

    while True:
		dst = _thinningIteration(dst, 0)
		dst = _thinningIteration(dst, 1)
		diff = np.absolute(dst - prev)
		prev = dst.copy()
		if np.sum(diff) == 0:
			break


    return dst * 255.

def findStrokes(source, output):
    thinned = thinning(source)

image = cv2.imread('./data/1.jpg', cv2.CV_8UC1)
# print img.isContinuous()
height, width = image.shape
original = image.copy()
v_original = image.copy()

cv2.bitwise_not(original, original)
original = cv2.dilate(original, np.ones((3, 3), np.uint8), iterations=2)
cv2.bitwise_not(original, original)

image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations= 1)
image = cv2.GaussianBlur(image, (9, 9), 0, 0.0, 4)


scaleFactor = IMAGE_HEIGHT / (height)
n_width = scaleFactor * width
image = cv2.resize(image, (int(n_width), int(IMAGE_HEIGHT)))
original = cv2.resize(original, (int(n_width), int(IMAGE_HEIGHT)))
v_original = cv2.resize(v_original, (int(n_width), int(IMAGE_HEIGHT)))


image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

thinned = thinning(image)
# print np.amin(thinned)
# print np.amax(thinned)
# exit()
##====================================================================================================




# cv2.namedWindow('image', cv2.WINDOW_NORMAL )
# th2 = cv2.resize(th2, (0,0), fx=0.80, fy=0.80)
cv2.imshow('image',image)
cv2.imshow('thinned',thinned)
cv2.imshow('original',v_original)
cv2.waitKey(0)
cv2.destroyAllWindows()
