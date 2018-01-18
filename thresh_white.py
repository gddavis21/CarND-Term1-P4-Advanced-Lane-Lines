import cv2
import numpy as np
import lanelines as LL

#optional argument
def nothing(x):
    pass

########################

# prsp = LL.PerspectiveTransform.make_top_down()
    
# cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
# cv2.createTrackbar('Krn', 'image', 25, 40, nothing)
# cv2.createTrackbar('Off', 'image', 33, 100, nothing)

# while(1):
    # frame = cv2.imread('./test_images/test6.jpg')
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # warp = prsp.warp_image(gray, (gray.shape[1], gray.shape[0]))

    # B = cv2.getTrackbarPos('Krn', 'image') *2 + 3
    # C = cv2.getTrackbarPos('Off', 'image')
    
    # mask = cv2.adaptiveThreshold(
        # warp, 
        # maxValue=255, 
        # adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
        # thresholdType=cv2.THRESH_BINARY,
        # blockSize=B,
        # C=-C)

    # cv2.imshow('image', mask)

    # k = cv2.waitKey(5) & 0xFF
    # if k == 27:
        # break

# cv2.destroyAllWindows()

#########################

cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Thr', 'image', 200, 255, nothing)

while(1):
    frame = cv2.imread('./test_images/test5.jpg')
    t = cv2.getTrackbarPos('Thr', 'image')
    thr_lower = np.array([t,t,t])
    thr_upper = np.array([255,255,255])
    seg = cv2.inRange(frame, thr_lower, thr_upper)
    cv2.imshow('image', seg)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
