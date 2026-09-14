"""
Cardboard Steering Wheel -> Keyboard Controller
Stick a RED marker and a BLUE marker on opposite sides of your cardboard
wheel's rim. Tracks both via HSV color thresholding, computes the wheel's
rotation angle, and simulates arrow key presses so any keyboard-controlled
browser game reads it as real steering input.

SETUP BEFORE RUNNING:
1. Put a red sticker/tape dot on one side of the wheel rim, blue on the
   opposite side (roughly 180 degrees apart).
2. Run this script, hold the wheel level (both markers roughly horizontal)
   with the camera facing you - this is your "centered" position.
3. Open your browser game in another window, click into it so it has
   keyboard focus, then start rotating the wheel.

Run: python steering_wheel.py
Quit: press 'q' (with the OpenCV window focused)

TUNING: if the color masks in the small preview windows look noisy/empty,
adjust the HSV ranges below for your lighting and marker colors.
"""

import cv2
import numpy as np
import keyboard
import math

# ---- HSV color ranges for the two markers. Tune these for your stickers. ----
# Red wraps around the HSV hue circle, so it needs two ranges.
RED_LOWER1, RED_UPPER1 = np.array([0, 120, 70]), np.array([10, 255, 255])
RED_LOWER2, RED_UPPER2 = np.array([170, 120, 70]), np.array([180, 255, 255])
BLUE_LOWER, BLUE_UPPER = np.array([100, 120, 70]), np.array([130, 255, 255])

MIN_CONTOUR_AREA = 150   # ignore tiny color noise blobs

# Steering zones in degrees. 0 = wheel level/centered.
DEAD_ZONE = 12       # +/- this many degrees counts as "centered", no key held
TURN_THRESHOLD = 12  # beyond dead zone, hold direction key

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

current_direction = None   # None, 'left', or 'right'


def find_marker_center(hsv_frame, lower, upper, lower2=None, upper2=None):
    mask = cv2.inRange(hsv_frame, lower, upper)
    if lower2 is not None:
        mask2 = cv2.inRange(hsv_frame, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def set_direction(new_direction):
    global current_direction
    if new_direction == current_direction:
        return

    if current_direction == "left":
        keyboard.release("left")
    elif current_direction == "right":
        keyboard.release("right")

    if new_direction == "left":
        keyboard.press("left")
    elif new_direction == "right":
        keyboard.press("right")

    current_direction = new_direction


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_center, red_mask = find_marker_center(hsv, RED_LOWER1, RED_UPPER1, RED_LOWER2, RED_UPPER2)
    blue_center, blue_mask = find_marker_center(hsv, BLUE_LOWER, BLUE_UPPER)

    angle_deg = None

    if red_center and blue_center:
        cv2.circle(frame, red_center, 8, (0, 0, 255), -1)
        cv2.circle(frame, blue_center, 8, (255, 0, 0), -1)
        cv2.line(frame, red_center, blue_center, (0, 255, 0), 2)

        dx = blue_center[0] - red_center[0]
        dy = blue_center[1] - red_center[1]
        angle_deg = math.degrees(math.atan2(dy, dx))

        # normalize so 0 degrees = horizontal, regardless of which marker
        # ends up left/right after rotation
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        if angle_deg > TURN_THRESHOLD:
            set_direction("right")
        elif angle_deg < -TURN_THRESHOLD:
            set_direction("left")
        elif abs(angle_deg) <= DEAD_ZONE:
            set_direction(None)
        # else: in the gap between dead zone and turn threshold, hold last state

    else:
        set_direction(None)   # markers lost, release keys for safety

    status = f"Angle: {angle_deg:.1f} deg" if angle_deg is not None else "Markers not detected"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    dir_text = f"Direction: {current_direction or 'center'}"
    cv2.putText(frame, dir_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Steering Wheel Controller", frame)
    cv2.imshow("Red mask (tune if noisy)", red_mask)
    cv2.imshow("Blue mask (tune if noisy)", blue_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

set_direction(None)   # release any held key on exit
cap.release()
cv2.destroyAllWindows()
