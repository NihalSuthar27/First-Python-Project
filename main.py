import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Initialize
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
click_cooldown = 0.3  # seconds
last_click_time = 0

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Important points
            index = lm_list[8]   # index tip
            thumb = lm_list[4]   # thumb tip
            middle = lm_list[12] # middle tip
            ring = lm_list[16]   # ring finger tip

            # ---------------- Mouse Movement ----------------
            screen_x = np.interp(index[0], (0, w), (0, screen_w))
            screen_y = np.interp(index[1], (0, h), (0, screen_h))
            pyautogui.moveTo(screen_x, screen_y)

            # ---------------- Left Click ----------------
            if distance(index, thumb) < 30:  
                if time.time() - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = time.time()

            # ---------------- Right Click ----------------
            if distance(ring, thumb) < 30:  
                if time.time() - last_click_time > click_cooldown:
                    pyautogui.click(button="right")
                    last_click_time = time.time()

            # ---------------- Scroll Up ----------------
            if distance(index, middle) < 30 and index[1] < middle[1]:
                pyautogui.scroll(50)  # up
                time.sleep(0.2)

            # ---------------- Scroll Down ----------------
            if distance(index, middle) < 30 and index[1] > middle[1]:
                pyautogui.scroll(-50)  # down
                time.sleep(0.2)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
