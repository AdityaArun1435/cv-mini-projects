"""
Webcam Darts - v2: Animated Throw, Colored Board, Smoother Tracking
Aim with your index fingertip, snap your hand forward fast (toward the
camera, like a real throw) to release. Tracking uses a short rolling
window instead of single-frame deltas, and factors in forward (depth)
motion, not just screen-plane speed, so a real throwing motion is
detected more reliably than incidental hand movement.

Run: python webcam_darts_v2.py
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
from collections import deque

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

# (radius, score, fill color BGR) outermost first
RINGS = [
    (200, 5,  (40, 90, 40)),     # dark green
    (160, 10, (225, 225, 225)),  # cream
    (120, 15, (40, 90, 40)),     # dark green
    (80,  20, (225, 225, 225)),  # cream
    (45,  25, (60, 60, 200)),    # red
    (18,  50, (30, 30, 130)),    # dark maroon bullseye
]
BACKING_RADIUS = RINGS[0][0] + 25
BACKING_COLOR = (35, 55, 70)   # dark wood-ish brown

# --- Tracking window ---
HISTORY_LEN = 6            # frames kept for smoothed velocity
Z_SCALE = 400              # scales mediapipe's small z units into pixel-ish units

# --- Throw detection tuning ---
THROW_SPEED_THRESHOLD = 700
COOLDOWN_SEC = 0.6

# --- Precision mapping ---
MIN_WOBBLE_PX = 8
MAX_WOBBLE_PX = 70
WOBBLE_REFERENCE_SPEED = 1800

# --- Dart flight animation ---
FLIGHT_DURATION = 0.35
ARC_HEIGHT = 40

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
    for radius, score, _ in RINGS:
        if dist <= radius:
            return score
    return 0


def wobble_for_speed(speed):
    ratio = min(speed / WOBBLE_REFERENCE_SPEED, 1.0)
    return MAX_WOBBLE_PX - ratio * (MAX_WOBBLE_PX - MIN_WOBBLE_PX)


def draw_board(frame):
    cv2.circle(frame, BOARD_CENTER, BACKING_RADIUS, BACKING_COLOR, -1)
    for radius, score, color in RINGS:
        cv2.circle(frame, BOARD_CENTER, radius, color, -1)
    for radius, score, _ in RINGS:
        cv2.circle(frame, BOARD_CENTER, radius, (20, 20, 20), 1)
        label_pos = (BOARD_CENTER[0] - 10, BOARD_CENTER[1] - radius + 14)
        cv2.putText(frame, str(score), label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)


def draw_dart(frame, tip_pos, angle_rad, length=22, tip_color=(255, 255, 255), tail_color=(0, 0, 255)):
    """Draws a simple dart icon: tip circle + shaft, pointing along angle_rad."""
    tip_x, tip_y = int(tip_pos[0]), int(tip_pos[1])
    tail_x = int(tip_x - length * math.cos(angle_rad))
    tail_y = int(tip_y - length * math.sin(angle_rad))
    cv2.line(frame, (tip_x, tip_y), (tail_x, tail_y), tail_color, 3)
    cv2.circle(frame, (tip_x, tip_y), 6, tip_color, -1)
    cv2.circle(frame, (tip_x, tip_y), 6, (0, 0, 0), 1)


high_score = load_high_score()
total_score = 0
throw_count = 0

# rolling history of (time, x, y, z)
history = deque(maxlen=HISTORY_LEN)

armed = False
peak_speed = 0
peak_pos = None
last_throw_time = -999
last_throw_message = ""

# active flight animation state
flight_active = False
flight_start_time = 0
flight_start_pos = None
flight_end_pos = None
flight_pending_score = 0

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
            z_val = lm[8].z * Z_SCALE   # more negative = closer to camera

        if fingertip is not None and not flight_active:
            aim_angle = math.atan2(BOARD_CENTER[1] - fingertip[1], BOARD_CENTER[0] - fingertip[0])
            draw_dart(frame, fingertip, aim_angle)
        history.append((now, fingertip[0], fingertip[1], z_val))

        if len(history) >= 2:
            t0, x0, y0, z0 = history[0]
            t1, x1, y1, z1 = history[-1]
            dt = t1 - t0
            if dt > 0.01:
                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt
                vz = (z1 - z0) / dt   # negative vz = moving toward camera (forward throw)
                # forward motion (toward camera) contributes to speed even if
                # x/y barely change, which is what a real throw looks like
                speed = math.sqrt(vx**2 + vy**2 + max(-vz, 0) ** 2)

        if speed >= THROW_SPEED_THRESHOLD:
            if not armed:
                armed = True
                peak_speed = speed
                peak_pos = fingertip
            elif speed > peak_speed:
                peak_speed = speed
                peak_pos = fingertip
        else:
            if armed and (now - last_throw_time) > COOLDOWN_SEC and not flight_active:
                wobble_std = wobble_for_speed(peak_speed)
                landed_x = peak_pos[0] + random.gauss(0, wobble_std)
                landed_y = peak_pos[1] + random.gauss(0, wobble_std)
                dist = math.hypot(landed_x - BOARD_CENTER[0], landed_y - BOARD_CENTER[1])
                points = score_for_distance(dist)

                flight_active = True
                flight_start_time = now
                flight_start_pos = peak_pos
                flight_end_pos = (landed_x, landed_y)
                flight_pending_score = points
                last_throw_time = now

            armed = False
            peak_speed = 0
            peak_pos = None
    else:
        history.clear()
        armed = False

    # --- animate dart flight ---
    if flight_active:
        r = (now - flight_start_time) / FLIGHT_DURATION
        if r >= 1.0:
            r = 1.0
            total_score += flight_pending_score
            throw_count += 1
            last_throw_message = f"Threw for {flight_pending_score} points! (speed {int(peak_speed)} px/s)"
            if total_score > high_score:
                high_score = total_score
                save_high_score(high_score)
            flight_active = False

        cur_x = flight_start_pos[0] + (flight_end_pos[0] - flight_start_pos[0]) * r
        cur_y = flight_start_pos[1] + (flight_end_pos[1] - flight_start_pos[1]) * r
        arc_offset = -ARC_HEIGHT * math.sin(math.pi * r)
        flight_angle = math.atan2(
            flight_end_pos[1] - flight_start_pos[1], flight_end_pos[0] - flight_start_pos[0])
        draw_dart(frame, (cur_x, cur_y + arc_offset), flight_angle)

        if r >= 1.0:
            cv2.circle(frame, (int(flight_end_pos[0]), int(flight_end_pos[1])), 5, (0, 0, 255), -1)

    cv2.putText(frame, f"Score: {total_score}  Throws: {throw_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"High score: {high_score}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Fingertip speed: {int(speed)} px/s", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, "Aim with fingertip, snap hand forward to throw",
                (10, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    if last_throw_message and now - last_throw_time < 1.5:
        cv2.putText(frame, last_throw_message, (FRAME_W // 2 - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Webcam Darts v2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nFinal score: {total_score} across {throw_count} throws")
print(f"High score on record: {high_score}")
