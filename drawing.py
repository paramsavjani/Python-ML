import math
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
        lmList = hand["lmList"]
        bbox = hand["bbox"]
        return fingers, lmList, bbox
    else:
        return None


def draw(info, previous_pos=None, canvas=None):
    fingers, lmList, bbox = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if previous_pos is None:
            previous_pos = current_pos
        cv2.line(canvas, previous_pos, current_pos, (255, 0, 255), 10)

    return current_pos, canvas


def erase(info, canvas, img):
    _, lmList, bbox = info

    center_x = int((lmList[15][0] + lmList[11][0]) / 2)
    center_y = int((lmList[15][1] + lmList[11][1]) / 2)
    center = (center_x, center_y)

    radius = int(
        math.sqrt(
            (lmList[5][0] - lmList[17][0]) ** 2 + (lmList[5][1] - lmList[17][1]) ** 2
        )
        / 1.5
    )

    cv2.circle(canvas, center, radius, (0, 0, 0), -1)
    cv2.circle(img, center, radius, (0, 255, 0), cv2.FILLED)

    return canvas


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
        fingers, lmList, bbox = info
        if fingers == [1, 1, 1, 1, 1] or fingers == [0, 1, 1, 1, 1]:
            canvas = erase(info, canvas, img)

        else:
            previous_pos, canvas = draw(info, previous_pos, canvas)

    image_combined = cv2.addWeighted(img, 0.6, canvas, 0.4, 0)

    cv2.imshow("Combined", image_combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
