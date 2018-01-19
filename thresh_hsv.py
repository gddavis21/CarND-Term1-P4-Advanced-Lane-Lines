import cv2
import numpy as np
#optional argument
def nothing(x):
    pass

cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Hue', 'image', 15, 169, nothing)
cv2.createTrackbar('Sat', 'image', 80, 255, nothing)
cv2.createTrackbar('Val', 'image', 120, 255, nothing)

while(1):
    frame = cv2.imread('./test_images/test5.jpg')
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = cv2.getTrackbarPos('Hue', 'image')
    s = cv2.getTrackbarPos('Sat', 'image')
    v = cv2.getTrackbarPos('Val', 'image')

    HSVLOW=np.array([h,s,v])
    HSVHIGH=np.array([h+10,255,255])

    #apply the range on a mask
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
    # res = cv2.bitwise_and(frame,frame, mask =mask)

    cv2.imshow('image', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()