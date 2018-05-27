from nms_cpu_ import *
import numpy as np

dets = np.array( [[204, 102, 358, 250, 0.5],
	[257, 118, 380, 250, 0.7],
	[280, 135, 400, 250, 0.6],
	[255, 118, 360, 235, 0.7]])
threshold = 0.3

keep = nms_cpu(dets,threshold)
print(keep)
