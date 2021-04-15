import cv2  # opencv-python
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)           # captures webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils     # used for drawing landmarks

# sets variables
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # changes to RGB
    results = hands.process(imgRGB)                         # processes RGB image by using process method
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:                # allows for multiple hands
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4:                                             # if point is 4 (change for different fingers - 4 is thumb)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)          # draws circle on point

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)    # draws landmarks and connections on image to output

    # method for calculating FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # outputs FPS to user
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1,
                (255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


