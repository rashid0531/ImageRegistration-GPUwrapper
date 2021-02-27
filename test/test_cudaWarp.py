
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import GpuWrapper as gw
import numpy as np

img = cv2.imread('sudokusmall.png')
rows,cols,ch = img.shape
  
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
 
M = cv2.getPerspectiveTransform(pts1,pts2)
img=np.float32(img)
M=np.float32(M)

print(img.ndim)
print(M.ndim)    
dst = gw.cudaWarpPerspectiveWrapper(img,M,(300,300),1)
A = gw.match_feature(img,M)
print(A)
#dst = cv2.warpPerspective(img,M,(300,300))
   
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.savefig('myfig')
