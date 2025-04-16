import cv2
import mediapipe as mp
import math
from pynput.keyboard import Key, Controller

keyboard = Controller()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)



"""
goal: 
walk forward = walk in place
turn the camera point left and right
jump forward = point forward
climb = climbing motionZ
emote = closed fist = run /e dance
zoom camera in = swim inward
zoom camera out = swim outward
"""

def getPointDirection(handmarks):
    origin = handmarks.landmark[5]
    tip = handmarks.landmark[8]
    xLength = tip.x-origin.x
    yLength = -(tip.y-origin.y)
    if xLength >= 0:
        keyboard.press(Key.right)
        keyboard.release(Key.right)
        return "Right"
    keyboard.press(Key.left)
    keyboard.release(Key.left)
    return "Left"
    # angle = math.degrees(math.atan2(yLength, xLength))
    # return str(round(angle % 360))
    
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = getPointDirection(hand_landmarks)
            if gesture:
                cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3)

    cv2.imshow("Pointing Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
