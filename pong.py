import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture(1)
img = cv2.imread('trasparency.jpg', 1)
font = cv2.FONT_HERSHEY_SIMPLEX


ay1 = 235
ay2 = 375

by1 = 235
by2 = 375
speed = 4

count1 = 0
count2 = 0

px1 = 390
px2 = 410
py1 = 290
py2 = 310

incx = speed
incy = -speed
conf = 0


def physics():
    global px1, px2, py1, py2, incx, incy, count1, count2, ay1, ay2, by1, by2, speed

    px1 += incx
    px2 += incx
    py1 += incy
    py2 += incy

    if px2 >= 800:
        px1 = 390
        px2 = 410
        py1 = 290
        py2 = 310
        incx = -speed
        count1 +=1

    elif px1 <= 0:
        px1 = 390
        px2 = 410
        py1 = 290
        py2 = 310
        incx = speed
        count2 +=1

    if py2 >= 600:
        incy = -speed
    if py1 <=55:
        incy = speed

    if px2 >= 750 and py1 >= by1 and py2 <= by2:
        incx = -speed
    if px1 <= 70 and py1 >= ay1 and py2 <= ay2:
        incx = speed

    if ay1 <= 55:
        ay1 = 55
        ay2 = 185
    if by1 <= 55:
        by1 = 55
        by2 = 185
    if ay2 >= 595:
        ay1 = 465
        ay2 = 595
    if by2 >= 595:
        by1 = 465
        by2 = 595

    
def two_players():
    global c1, M1, cx1, cy1, ay1, ay2, c2, M2, cx2, cy2, by1, by2, conf, pos_ball, pos_palette, error, final_pos

    jugador1 = mask[0:550, 0:400]
    jugador2 = mask[0:550, 400:800]
    contours1, _ = cv2.findContours(jugador1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(jugador2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours1) > 0:
        c1 = max(contours1, key=cv2.contourArea)
        M1 = cv2.moments(c1)
        cx1 = int(M1['m10']/M1['m00'])
        cy1 = int(M1['m01']/M1['m00'])
        ay1 = cy1-65
        ay2 = cy1+65
    
    if len(contours2) > 0:
        c2 = max(contours2, key=cv2.contourArea)
        M2 = cv2.moments(c2)
        cx2 = int(M2['m10']/M2['m00'])
        cy2 = int(M2['m01']/M2['m00'])
        by1 = cy2-65
        by2 = cy2+65

while (1):
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=800)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([124, 98, 90])
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # if frame.shape != img.shape:
    # # Resize img to match frame
    #     img = cv2.resize(img, (frame.shape[1], frame.shape[0]))

    # # Check the number of channels
    # if len(frame.shape) != len(img.shape):
    #     # Convert img to match frame's number of channels
    #     if len(frame.shape) == 3:  # frame is a color image
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     else:  # frame is a grayscale image
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # frame = cv2.addWeighted(frame, 0.8, img, 0.2, 0)

    cv2.rectangle(frame, (50, ay1), (70, ay2), (254, 112, 68), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (750, by1), (770, by2), (33, 49, 255), -1, cv2.LINE_AA)

    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), -1, cv2.LINE_AA)

    cv2.rectangle(frame, (397, 0), (403, 600), (224, 254, 253), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 50), (800, 55), (224, 254, 253), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 595), (800, 600), (224, 254, 253), -1, cv2.LINE_AA)

    cv2.putText(frame, " Player 1 Score: " + str(count1) + "         Player 2 Score:  " + str(count2), (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (132, 255, 184), 2, cv2.LINE_AA)

    two_players()

    physics()


# cv2.destroyAllWindows()
# cap.release()

