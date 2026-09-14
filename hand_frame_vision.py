"""
Hand-Frame Vision Effect
Make an L-shape/rectangle with both hands (thumb + index finger extended on
each hand) and the region inside gets a thermal/night-vision-style effect
applied, rest of the frame stays normal.

Run: python hand_frame_vision.py
Quit: press 'q'
Switch effect: press 'e' to cycle through effect modes
"""

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

MIN_FRAME_SIZE = 60   # minimum width/height in pixels to count as a valid frame
EFFECTS = ["thermal", "edge_xray", "night_vision"]
effect_idx = 0


def apply_effect(roi, mode):
    if roi.size == 0:
        return roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if mode == "thermal":
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    if mode == "edge_xray":
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # tint edges cyan for an x-ray look
        edges_colored[:, :, 0] = edges_colored[:, :, 0]
        edges_colored[:, :, 1] = edges_colored[:, :, 1]
        edges_colored[:, :, 2] = 0
        return edges_colored

    if mode == "night_vision":
        green = np.zeros_like(roi)
        green[:, :, 1] = cv2.equalizeHist(gray)
        return green

    return roi


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        points = []
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            # thumb tip (4) and index tip (8) from each hand
            points.append((int(lm[4].x * w), int(lm[4].y * h)))
            points.append((int(lm[8].x * w), int(lm[8].y * h)))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = max(min(xs), 0), min(max(xs), w)
        y_min, y_max = max(min(ys), 0), min(max(ys), h)

        if (x_max - x_min) > MIN_FRAME_SIZE and (y_max - y_min) > MIN_FRAME_SIZE:
            roi = frame[y_min:y_max, x_min:x_max]
            effect_roi = apply_effect(roi, EFFECTS[effect_idx])
            frame[y_min:y_max, x_min:x_max] = effect_roi
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    cv2.putText(frame, f"Effect: {EFFECTS[effect_idx]}  (press 'e' to switch)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Hand-Frame Vision", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('e'):
        effect_idx = (effect_idx + 1) % len(EFFECTS)

cap.release()
cv2.destroyAllWindows()
