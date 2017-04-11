import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

cv2.ocl.setUseOpenCL(False)

MIN_MATCH_COUNT = 10

img1 = cv2.imread(str(sys.argv[1]),0)  # queryImage
img2 = cv2.imread(str(sys.argv[2]),0)   # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
	if m.distance < 0.7*n.distance:
		good.append(m)

if len(good)>MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good[:50] ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good[:50] ]).reshape(-1,1,2)
	
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()
	
	if len(img1.shape) > 2:
		h,w,d = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)
	
		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
	
else:
	print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
	matchesMask = None

draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)   
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:50],None,**draw_params)
img4 = cv2.drawKeypoints(img1,kp1,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("sift_Matching.jpg",img3)
cv2.imwrite("sift_Matching_KeyPoints.jpg",img4)
	
