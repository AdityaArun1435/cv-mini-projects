"""
Hand Gesture Volume Control
Tracks thumb-index finger distance via MediaPipe hand landmarks and maps it
to system volume in real time.

Run: python gesture_volume_control.py
Quit: press 'q'
"""

import cv2
import mediapipe as mp
import numpy as np
import math

# Try to hook into Windows system volume. On Mac/Linux this will fail
# silently and the script still runs, just without actually changing
# the OS volume (the on-screen bar and printed % still work).
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range = volume_ctrl.GetVolumeRange()
    MIN_VOL, MAX_VOL = vol_range[0], vol_range[1]
    SYSTEM_VOLUME_AVAILABLE = True
except Exception:
    SYSTEM_VOLUME_AVAILABLE = False
    MIN_VOL, MAX_VOL = -65.0, 0.0

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

# Calibration range for thumb-index pixel distance -> volume mapping.
# Adjust these two if your camera/hand size makes the bar clip at the ends.
DIST_MIN, DIST_MAX = 25, 220

# Lock toggle uses middle, ring, and pinky only — completely separate from
# thumb-index, which is what drives the volume. Extend those three fingers
# to adjust volume normally; curl all three together to toggle the lock.
# This keeps the lock gesture from ever touching the volume value.
LOCK_TIPS = [12, 16, 20]   # middle, ring, pinky tips
LOCK_PIPS = [10, 14, 18]   # corresponding lower joints

is_locked = False
locked_vol_percent = 0
lock_gesture_frame_count = 0
LOCK_HOLD_FRAMES = 8   # frames the curl must be held before toggling, debounce

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    vol_percent = 0
    length = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            x1, y1 = int(lm[4].x * w), int(lm[4].y * h)   # thumb tip
            x2, y2 = int(lm[8].x * w), int(lm[8].y * h)   # index tip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Lock gesture: middle, ring, and pinky all curled together.
            # Thumb and index are ignored here so this never fights with
            # the pinch-based volume reading.
            lock_curled = sum(
                1 for tip, pip in zip(LOCK_TIPS, LOCK_PIPS)
                if lm[tip].y > lm[pip].y
            )
            is_lock_gesture = lock_curled == 3

            if is_lock_gesture:
                lock_gesture_frame_count += 1
            else:
                lock_gesture_frame_count = 0

            if lock_gesture_frame_count == LOCK_HOLD_FRAMES:
                is_locked = not is_locked

            if is_locked or is_lock_gesture:
                # Frozen: either locked, or currently mid-gesture (don't let
                # the pinch distance leak through while curling fingers).
                vol_percent = locked_vol_percent
            else:
                vol_percent = np.interp(length, [DIST_MIN, DIST_MAX], [0, 100])
                vol_percent = float(np.clip(vol_percent, 0, 100))
                locked_vol_percent = vol_percent

                if SYSTEM_VOLUME_AVAILABLE:
                    vol_scalar = np.interp(vol_percent, [0, 100], [MIN_VOL, MAX_VOL])
                    volume_ctrl.SetMasterVolumeLevel(vol_scalar, None)

                if length < DIST_MIN:
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

    # On-screen volume bar
    bar_y = int(np.interp(vol_percent, [0, 100], [400, 150]))
    cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 3)
    cv2.rectangle(frame, (50, bar_y), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, f'{int(vol_percent)} %', (40, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    status = "System volume: LIVE" if SYSTEM_VOLUME_AVAILABLE else "System volume: not hooked (non-Windows)"
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    lock_text = "LOCKED (curl middle/ring/pinky to unlock)" if is_locked else "Curl middle/ring/pinky to lock volume"
    lock_color = (0, 0, 255) if is_locked else (0, 200, 0)
    cv2.putText(frame, lock_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_color, 2)

    cv2.imshow("Gesture Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
