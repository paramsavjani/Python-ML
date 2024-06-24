from asyncio import sleep
import cvzone.FaceMeshModule as fm
import cv2
import cvzone
from cvzone.PlotModule import LivePlot


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./video.mp4")
detector = fm.FaceMeshDetector(maxFaces=2)
plotY = LivePlot(640, 480, [20, 50], invert=True, char=" ")


newidList = [7, 33, 157, 158, 159, 160, 161, 163, 144, 145, 155, 153, 154, 173, 133]

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243, 173, 133]


ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)


while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]

        # for id in newidList:
        #     cv2.circle(img, face[id], 6, (0, 255, 0), cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        ratio = int((lenghtVer / lenghtHor) * 100)

        ratioList.append(ratio)

        if len(ratioList) > 2:
            ratioList.pop(0)

        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            color = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)

        cvzone.putTextRect(img, f"Blink Count: {blinkCounter}", (50, 100), colorR=color)

        imgPlot = plotY.update(ratio, color)
        img = cv2.resize(img, (640, 400))
        imgStack = cvzone.stackImages([img, imgPlot], 1, 1)
    else:
        img = cv2.resize(img, (640, 400))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    cv2.waitKey(25)
