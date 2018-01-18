import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass

cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Sat', 'image', 70, 255, nothing)
cv2.createTrackbar('Val', 'image', 130, 255, nothing)

while(1):
    frame = cv2.imread('./test_images/test6.jpg')
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    s = cv2.getTrackbarPos('Sat', 'image')
    v = cv2.getTrackbarPos('Val', 'image')

    HSVLOW=np.array([15,s,v])
    HSVHIGH=np.array([25,255,255])

    #apply the range on a mask
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)

    cv2.imshow('image', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()

