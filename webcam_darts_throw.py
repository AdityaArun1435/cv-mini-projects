"""
Webcam Darts - Motion-Throw Edition
Aim with your index fingertip. Snap your hand forward fast to "release" a
throw - wherever your fingertip was at release is where it aims, and how
fast you snapped it determines precision (fast = tighter, slow = wobblier).

Run: python webcam_darts_throw.py
Quit: press 'q'
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import json
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

FRAME_W, FRAME_H = 960, 720
BOARD_CENTER = (FRAME_W // 2, FRAME_H // 2)

RINGS = [
    (200, 5),
    (160, 10),
    (120, 15),
    (80, 20),
    (45, 25),
    (18, 50),   # bullseye
]

# --- Throw detection tuning ---
# Speed here is in pixels/second, measured on the index fingertip.
THROW_SPEED_THRESHOLD = 900   # must exceed this to count as a "throw" swing
COOLDOWN_SEC = 0.6            # minimum time between two separate throws

# --- Precision mapping ---
# Faster release = tighter grouping. These bound the wobble std dev.
MIN_WOBBLE_PX = 8      # wobble at very high speed (best case)
MAX_WOBBLE_PX = 70     # wobble right at the threshold speed (worst case)
WOBBLE_REFERENCE_SPEED = 2200   # speed at/above which wobble bottoms out

SAVE_FILE = "darts_highscore.json"


def load_high_score():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE) as f:
                return json.load(f).get("high_score", 0)
        except Exception:
            return 0
    return 0


def save_high_score(score):
    with open(SAVE_FILE, "w") as f:
        json.dump({"high_score": score}, f)


def score_for_distance(dist):
    for radius, score in RINGS:
        if dist <= radius:
            return score
    return 0


def wobble_for_speed(speed):
    ratio = min(speed / WOBBLE_REFERENCE_SPEED, 1.0)
    return MAX_WOBBLE_PX - ratio * (MAX_WOBBLE_PX - MIN_WOBBLE_PX)


def draw_board(frame):
    for radius, score in RINGS:
        cv2.circle(frame, BOARD_CENTER, radius, (255, 255, 255), 2)
        label_pos = (BOARD_CENTER[0] + radius - 15, BOARD_CENTER[1] - 5)
        cv2.putText(frame, str(score), label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


high_score = load_high_score()
total_score = 0
throw_count = 0

prev_pos = None
prev_time = None
armed = False               # True once speed crosses threshold, until it drops back down
peak_speed = 0
peak_pos = None
last_throw_time = -999
last_throw_message = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_board(frame)

    now = time.time()
    speed = 0
    fingertip = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            fingertip = (int(lm[8].x * FRAME_W), int(lm[8].y * FRAME_H))

        cv2.circle(frame, fingertip, 8, (0, 255, 255), -1)

        if prev_pos is not None and prev_time is not None:
            dt = now - prev_time
            if dt > 0:
                dist_moved = math.hypot(fingertip[0] - prev_pos[0], fingertip[1] - prev_pos[1])
                speed = dist_moved / dt

        prev_pos = fingertip
        prev_time = now

        if speed >= THROW_SPEED_THRESHOLD:
            if not armed:
                armed = True
                peak_speed = speed
                peak_pos = fingertip
            elif speed > peak_speed:
                peak_speed = speed
                peak_pos = fingertip
        else:
            if armed and (now - last_throw_time) > COOLDOWN_SEC:
                # swing just ended - finalize the throw using recorded peak
                wobble_std = wobble_for_speed(peak_speed)
                landed_x = peak_pos[0] + random.gauss(0, wobble_std)
                landed_y = peak_pos[1] + random.gauss(0, wobble_std)
                dist = math.hypot(landed_x - BOARD_CENTER[0], landed_y - BOARD_CENTER[1])
                points = score_for_distance(dist)

                total_score += points
                throw_count += 1
                last_throw_message = f"Threw for {points} points! (speed {int(peak_speed)} px/s)"
                last_throw_time = now

                if total_score > high_score:
                    high_score = total_score
                    save_high_score(high_score)

                cv2.circle(frame, (int(landed_x), int(landed_y)), 8, (0, 0, 255), -1)

            armed = False
            peak_speed = 0
            peak_pos = None
    else:
        prev_pos = None
        prev_time = None
        armed = False

    cv2.putText(frame, f"Score: {total_score}  Throws: {throw_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"High score: {high_score}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Fingertip speed: {int(speed)} px/s", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, "Aim with fingertip, snap hand forward to throw",
                (10, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    if last_throw_message and now - last_throw_time < 1.2:
        cv2.putText(frame, last_throw_message, (FRAME_W // 2 - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Webcam Darts - Motion Throw", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nFinal score: {total_score} across {throw_count} throws")
print(f"High score on record: {high_score}")
