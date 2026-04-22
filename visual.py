import cv2
import mediapipe as mp
import numpy as np

# =========================
# Virtual Air Drawing App
# Features:
# - Hand tracking (MediaPipe)
# - Drawing with index finger
# - Color palette UI
# - Eraser mode
# - Clear canvas button
# =========================

# ---------- Setup ----------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas for drawing
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Default settings
prev_x, prev_y = 0, 0
brush_color = (255, 0, 0)  # Blue
brush_thickness = 5
eraser_thickness = 50
mode = "DRAW"  # DRAW / ERASE

# ---------- UI Button Positions ----------
buttons = {
    "RED": (50, 10, 150, 80),
    "GREEN": (170, 10, 270, 80),
    "BLUE": (290, 10, 390, 80),
    "ERASER": (410, 10, 560, 80),
    "CLEAR": (580, 10, 700, 80)
}

# ---------- Helper Function ----------
def draw_buttons(img):
    for key, (x1, y1, x2, y2) in buttons.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, key, (x1 + 10, y1 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# ---------- Main Loop ----------
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_buttons(img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []

            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Draw landmarks (optional)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if lm_list:
                index_x, index_y = lm_list[8]
                middle_x, middle_y = lm_list[12]

                # ---------- Gesture: Index Finger Up ----------
                if lm_list[8][1] < lm_list[6][1]:

                    # ---------- Check UI button clicks ----------
                    for key, (x1, y1, x2, y2) in buttons.items():
                        if x1 < index_x < x2 and y1 < index_y < y2:
                            if key == "RED":
                                brush_color = (0, 0, 255)
                                mode = "DRAW"
                            elif key == "GREEN":
                                brush_color = (0, 255, 0)
                                mode = "DRAW"
                            elif key == "BLUE":
                                brush_color = (255, 0, 0)
                                mode = "DRAW"
                            elif key == "ERASER":
                                mode = "ERASE"
                            elif key == "CLEAR":
                                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

                    # ---------- Drawing ----------
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = index_x, index_y

                    if mode == "DRAW":
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), 
                                 brush_color, brush_thickness)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), 
                                 (0, 0, 0), eraser_thickness)

                    prev_x, prev_y = index_x, index_y

                else:
                    prev_x, prev_y = 0, 0

    # ---------- Merge Canvas with Camera ----------
    img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    cv2.imshow("Virtual Drawing App", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
