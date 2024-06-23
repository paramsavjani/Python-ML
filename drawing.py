import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np


hcam = 1200
wcam = 1080

cap = cv2.VideoCapture(0)
cap.set(4, hcam)
cap.set(3, wcam)

detector = HandDetector(detectionCon=0.7, maxHands=1)


def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False)

    if hands:

        hand = hands[0]

        fingers = detector.fingersUp(hand)
        print(fingers)
        lmList = hand["lmList"]
        return fingers, lmList
    else:
        return None


def draw(info, previous_pos=None, canvas=None):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if previous_pos is None:
            previous_pos = current_pos
        cv2.line(canvas, previous_pos, current_pos, (255, 0, 255), 10)

    if fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(canvas)

    return current_pos, canvas


previous_pos = None
canvas = None
image_combined = None

while True:
    success, img = cap.read()

    if canvas is None:
        canvas = np.zeros_like(img)
        image_combined = img.copy()

    img = cv2.flip(img, 1)
    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        previous_pos, canvas = draw(info, previous_pos, canvas)

    image_combined = cv2.addWeighted(img, 0.6, canvas, 0.4, 0)

    cv2.imshow("Combined", image_combined)
    cv2.waitKey(1)
