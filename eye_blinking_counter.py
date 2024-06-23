from asyncio import sleep
import cvzone.FaceMeshModule as fm
import cv2

cap = cv2.VideoCapture("./video.mp4")
detector = fm.FaceMeshDetector(maxFaces=2)
idList = [7, 33, 157, 158, 159, 160, 161, 163, 144, 145, 155, 153, 154, 173, 133]

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)
    if faces[0]:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 6, (0, 255, 0), cv2.FILLED)

        leftUp = face[159]
        leftDown = face[145]
        leftLeft = face[130]
        leftRight = face[243]

        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
        
        

    cv2.imshow("Image", img)
    cv2.waitKey(100000)
