import cv2
import mediapipe as mp
import math
from pynput.keyboard import Key, Controller
import time

keyboard = Controller()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)
wristPos = {"Left":[0,0,0],
                "Right":[0,0,0]}

# def getPointDirection(handmarks):
#     origin = handmarks.landmark[5]
#     tip = handmarks.landmark[8]
#     xLength = tip.x-origin.x
#     yLength = -(tip.y-origin.y)
#     if xLength >= 0:
#         keyboard.press(Key.right)
#         keyboard.release(Key.right)
#         return "Right"
#     keyboard.press(Key.left)
#     keyboard.release(Key.left)
#     return "Left"
#     # angle = math.degrees(math.atan2(yLength, xLength))
#     # return str(round(angle % 360))

def isFist(handmarks):
    tips = [8, 12, 16, 20]
    bases = [5, 9, 13, 17]

    folded_fingers = 0
    open = 0
    for tip_id, base_id in zip(tips, bases):
        tip = handmarks.landmark[tip_id]
        base = handmarks.landmark[base_id]
        if tip.y > base.y:
            folded_fingers += 1
        else:
            open += 1

    if(folded_fingers >= 4):
        return "fist" 
    if(open >= 4):
        return "open hand"
    return "None"

def getHandSide(hand_label):
        return hand_label.classification[0].label
        
def mainControl(multiHandResults,handmarkResults,frame):
    isStarPunch = False
    for hand_landmarks, hand_label in zip(handmarkResults, multiHandResults):
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if isFist(hand_landmarks)== "open hand":
            # isStarPunch = True
            continue
        if isFist(hand_landmarks)== "None":
            continue
        wrist = hand_landmarks.landmark[0]
        wristPos[getHandSide(hand_label)][0] = wrist.x
        wristPos[getHandSide(hand_label)][1] = wrist.y
        wristPos[getHandSide(hand_label)][2] = wrist.z
    if(isStarPunch):
        cv2.putText(frame, "star punch", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        keyboard.press(Key.enter)
        time.sleep(0.1)
        keyboard.release(Key.enter)
        return
    xDist = abs(wristPos["Left"][0]-wristPos["Right"][0])*10
    yDist = abs(wristPos["Left"][1]-wristPos["Right"][1])*10
    dist = math.hypot(xDist,yDist)
    if dist <= 2 and yDist <= 1:
        avgXPos = (wristPos["Left"][0]+wristPos["Right"][0])*5
        if(avgXPos <= 3):
            cv2.putText(frame, "dodge left", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            keyboard.press("a")
            time.sleep(0.1)
            keyboard.release("a")
        if(3 <= avgXPos and avgXPos <= 6):
             cv2.putText(frame, "block", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
             keyboard.press("s")
             time.sleep(0.1)
             keyboard.release("s")
        if(avgXPos >= 6):
             cv2.putText(frame, "dodge right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
             keyboard.press("d")
             time.sleep(0.1)
             keyboard.release("d")
        return


    if wristPos["Left"][1] < wristPos["Right"][1]:
        y_value = wristPos["Left"][1]
        if y_value*10<=5:
            keyboard.press("w")
            keyboard.press("k")
            time.sleep(0.1)
            keyboard.release("w")
            keyboard.release("k")
            cv2.putText(frame, "left uppercut", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            return
    else:
        y_value = wristPos["Right"][1]
        if y_value*10<=5:
            keyboard.press("w")
            keyboard.press("l")
            time.sleep(0.1)
            keyboard.release("w")
            keyboard.release("l")
            cv2.putText(frame, "right uppercut", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            return

    if wristPos["Left"][2] < wristPos["Right"][2]:
            keyboard.press("k")
            time.sleep(0.1)
            keyboard.release("k")
            cv2.putText(frame, "left jab", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        keyboard.press("l")
        time.sleep(0.1)
        keyboard.release("l")
        cv2.putText(frame, "right jab", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
         
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        gesture = mainControl(results.multi_handedness,results.multi_hand_landmarks,frame)

    cv2.imshow("punchout hand control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
