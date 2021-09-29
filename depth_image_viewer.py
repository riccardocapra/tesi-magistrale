import numpy as np
import cv2 as cv
vis = np.zeros((384, 836), np.float32)
h,w = vis.shape
vis2 = cv.CreateMat(h, w, cv.CV_32FC3)
vis0 = cv.fromarray(vis)
cv.CvtColor(vis0, vis2, cv.CV_GRAY2BGR)