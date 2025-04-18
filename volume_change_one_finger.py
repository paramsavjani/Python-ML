import math
import cv2
import numpy as np
import time
import HandTrackingModule_My as htm

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Camera setup
wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

# Audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
minVol, maxVol = volume.GetVolumeRange()[:2]

# Detector
detector = htm.HandDetector(detectionCon=0.8)
pTime = 0
volSmooth = 0
lastVolPer = 0
frameCount = 0

# Calibration
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coeff = np.polyfit(x, y, 2)


def getDistanceCM(lmList):
    x1, y1 = lmList[5][1], lmList[5][2]
    x2, y2 = lmList[17][1], lmList[17][2]
    dist = int(math.hypot(x2 - x1, y2 - y1))
    A, B, C = coeff
    return A * dist ** 2 + B * dist + C


def drawBar(img, volPer):
    barHeight = int(np.interp(volPer, [0, 100], [400, 150]))
    color = (int(255 - volPer * 2.55), int(volPer * 2.55), 100)

    cv2.rectangle(img, (50, 150), (85, 400), (100, 100, 100), 3)
    cv2.rectangle(img, (50, barHeight), (85, 400), color, cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (45, 430), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    frameCount += 1

    if len(lmList) > 0:
        if len(lmList[0]) > 8:
            x1, y1 = lmList[0][4][1], lmList[0][4][2]
            x2, y2 = lmList[0][8][1], lmList[0][8][2]

            length = math.hypot(x2 - x1, y2 - y1)
            distCM = getDistanceCM(lmList[0])
            adjustedLength = distCM * distCM * length / 3200

            volTarget = np.interp(adjustedLength, [35, 220], [minVol, maxVol])
            volSmooth = volSmooth * 0.9 + volTarget * 0.1
            volPer = np.interp(volSmooth, [minVol, maxVol], [0, 100])

            fingers = detector.fingersUp()
            # Apply volume only if only thumb and index are up
            if fingers and fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
                # Change only if difference is significant (3%)
                if abs(volPer - lastVolPer) > 3 and frameCount % 3 == 0:
                    volume.SetMasterVolumeLevel(volSmooth, None)
                    lastVolPer = volPer

                drawBar(img, volPer)
                cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 150, 0), 3)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-6)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (480, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
