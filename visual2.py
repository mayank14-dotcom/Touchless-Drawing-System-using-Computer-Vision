import cv2
import mediapipe as mp
import numpy as np
import time

# ================================
# INIT
# ================================
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get camera size dynamically (IMPORTANT FIX)
ret, frame = cap.read()
h, w, c = frame.shape

# Canvas SAME size as camera
canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
brush_color = (255, 0, 0)
brush_thickness = 5
mode = "DRAW"

# ================================
# TOOLBAR
# ================================
tools = {
    "RED": (50, 20, 150, 100),
    "GREEN": (170, 20, 270, 100),
    "BLUE": (290, 20, 390, 100),
    "ERASE": (410, 20, 560, 100),
    "CLEAR": (580, 20, 730, 100),
    "SAVE": (750, 20, 880, 100)
}

def draw_toolbar(img):
    for key, (x1, y1, x2, y2) in tools.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, key, (x1+10, y1+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def save_canvas():
    filename = f"drawing_{int(time.time())}.png"
    cv2.imwrite(filename, canvas)
    print("Saved:", filename)

# ================================
# FULLSCREEN WINDOW
# ================================
cv2.namedWindow("MS PAINT AI", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("MS PAINT AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ================================
# MAIN LOOP
# ================================
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 🔥 FIX: resize image to canvas size (CRITICAL FIX)
    img = cv2.resize(img, (w, h))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_toolbar(img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            lm = []
            for id, lm_point in enumerate(handLms.landmark):
                cx, cy = int(lm_point.x * w), int(lm_point.y * h)
                lm.append((cx, cy))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if lm:

                index_x, index_y = lm[8]

                # ================================
                # GESTURE: Index finger up
                # ================================
                if lm[8][1] < lm[6][1]:

                    # ---------------- UI CLICK ----------------
                    for key, (x1,y1,x2,y2) in tools.items():
                        if x1 < index_x < x2 and y1 < index_y < y2:

                            if key == "RED":
                                brush_color = (0,0,255)
                                mode = "DRAW"

                            elif key == "GREEN":
                                brush_color = (0,255,0)
                                mode = "DRAW"

                            elif key == "BLUE":
                                brush_color = (255,0,0)
                                mode = "DRAW"

                            elif key == "ERASE":
                                mode = "ERASE"

                            elif key == "CLEAR":
                                canvas = np.zeros((h, w, 3), dtype=np.uint8)

                            elif key == "SAVE":
                                save_canvas()

                    # ---------------- DRAWING ----------------
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = index_x, index_y

                    if mode == "DRAW":
                        cv2.line(canvas, (prev_x, prev_y),
                                 (index_x, index_y),
                                 brush_color, brush_thickness)
                    else:
                        cv2.line(canvas, (prev_x, prev_y),
                                 (index_x, index_y),
                                 (0,0,0), 50)

                    prev_x, prev_y = index_x, index_y

                else:
                    prev_x, prev_y = 0, 0

    # ================================
    # FIXED MERGE (NO SIZE ERROR NOW)
    # ================================
    output = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    cv2.imshow("MS PAINT AI", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()