import cv2
import numpy as np
import matplotlib.pyplot as plt
from lanelines import Camera

camera = Camera()
camera.set_perspective(
    src_quad=np.float32([[310,648],[604,446],[678,446],[990,648]]), 
    dst_quad=np.float32([[420,719],[420,20],[840,20],[840,719]]))

img = cv2.imread('./test_images/straight_lines1_undist.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
warp = camera.warp_image(img, (img.shape[1], img.shape[0]))
cv2.imwrite('./test_images/straight_lines1_warped.jpg', warp)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
fig.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(warp)
ax2.set_title('Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
