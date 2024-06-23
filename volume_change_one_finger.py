import math
import cv2
import numpy as np
import time
import HandTrackingModule_My as htm

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


hcam = 640
wcam = 480


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minvol = volRange[0]
maxvol = volRange[1]


cap = cv2.VideoCapture(0)
cap.set(3, hcam)
cap.set(4, wcam)
pTime = 0

detelctor = htm.HandDetector(detectionCon=0.7)

# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C


def distance(lmList):

    x1, y1 = lmList[5][1], lmList[5][2]
    x2, y2 = lmList[17][1], lmList[17][2]

    distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
    A, B, C = coff
    distanceCM = A * distance**2 + B * distance + C
    return distanceCM


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detelctor.findHands(img)
    lmList = detelctor.findPosition(img, draw=False)

    if len(lmList) > 0:
        if len(lmList[0]) > 8:

            x1, y1 = lmList[0][4][1], lmList[0][4][2]
            x2, y2 = lmList[0][8][1], lmList[0][8][2]

            length = math.hypot(x2 - x1, y2 - y1)

            # here we want to convert the length to cm by distance function

            distance_frame = distance(lmList[0])
            length = distance_frame * distance_frame * length / 3200

            if length < 35:
                cv2.circle(
                    img,
                    (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    16,
                    (0, 0, 255),
                    cv2.FILLED,
                )
            else:
                cv2.circle(img, (x1, y1), 12, (37, 231, 41), cv2.FILLED)
                cv2.circle(img, (x2, y2), 12, (37, 231, 41), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (37, 231, 41), 3)

            # volume range -144 to 0
            # our hand range is 25 to 250
            print(length)
            vol = np.interp(length, [35, 220], [-55, maxvol])
            if vol <= -54:
                vol = minvol
            fingers = detelctor.fingersUp()
            if fingers[4] == 0:
                volume.SetMasterVolumeLevel(vol, None)

            cv2.putText(
                img,
                f"Volume: {int(np.interp(length, [25, 220], [0, 100]))}%",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(
                img,
                (50, int(np.interp(length, [25, 220], [400, 150]))),
                (85, 400),
                (0, 255, 0),
                cv2.FILLED,
            )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img, f"FPS: {int(fps)}", (500, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3
    )
    cv2.imshow("Video", img)
    cv2.waitKey(1)


# volume.GetMute()
# volume.GetMasterVolumeLevel()
# volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
