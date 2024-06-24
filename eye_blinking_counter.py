import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from collections import deque


cap = cv2.VideoCapture(0)



detector = FaceMeshDetector(maxFaces=1)


plotY = LivePlot(640, 480, [20, 50], invert=True, char=" ")


idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243, 173, 133]


ratioList = deque(maxlen=2)
blinkCounter = 0
color = (255, 0, 255)
flag = False
blinkThreshold = 35  


def calculate_eye_aspect_ratio(face):
    leftUp = face[159]
    leftDown = face[23]
    leftLeft = face[130]
    leftRight = face[243]

    lengthVer, _ = detector.findDistance(leftUp, leftDown)
    lengthHor, _ = detector.findDistance(leftLeft, leftRight)

    return (lengthVer / lengthHor) * 100


while True:
    
    # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img = cv2.flip(img, 1)

    
    img, faces = detector.findFaceMesh(img, draw=False)
    
    if faces:
        face = faces[0]

        
        ratio = calculate_eye_aspect_ratio(face)
        ratioList.append(ratio)
        ratioAvg = sum(ratioList) / len(ratioList)

        
        if ratioAvg < blinkThreshold and not flag:
            flag = True
            color = (0, 200, 0)
        elif ratioAvg >= blinkThreshold and flag:
            flag = False
            blinkCounter += 1
            color = (200, 0, 0)

        
        cvzone.putTextRect(img, f"Blink Count: {blinkCounter}", (40, 100), colorR=color)

        
        imgPlot = plotY.update(ratio, color)
        img = cv2.resize(img, (640, 400))
        imgStack = cvzone.stackImages([img, imgPlot], 1, 1)
    else:
        img = cv2.resize(img, (640, 400))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    
    cv2.imshow("Image", imgStack)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
