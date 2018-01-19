import cv2
import numpy as np
import lanelines as LL
#optional argument
def nothing(x):
    pass

cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Thr', 'image', 25, 50, nothing)
cv2.createTrackbar('Krn', 'image', 5, 10, nothing)

while(1):
    frame = cv2.imread('./test_images/test5.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t = cv2.getTrackbarPos('Thr', 'image')
    k = cv2.getTrackbarPos('Krn', 'image') *2 + 3
    
    grad_mag, grad_dir = LL.image_gradient(gray, kernel_size=k)

    #apply the range on a mask
    mask_mag = cv2.inRange(grad_mag, t, 255)
    mask_dir1 = cv2.inRange(grad_dir, 80, 100)
    mask_dir2 = cv2.inRange(grad_dir, 260, 280)
    mask_dir = cv2.bitwise_not(cv2.bitwise_or(mask_dir1, mask_dir2))
    mask = cv2.bitwise_and(mask_mag, mask_dir)
    # res = cv2.bitwise_and(frame,frame, mask =mask)

    cv2.imshow('image', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()