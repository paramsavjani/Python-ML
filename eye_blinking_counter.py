import cvzone.FaceMeshModule as fm
import cv2

cap = cv2.VideoCapture(0)
detector = fm.FaceMeshDetector(maxFaces=2)

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)
    print(faces)

    cv2.imshow("Image", img)
    cv2.waitKey(10000)
