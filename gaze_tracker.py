"""
Interview Coach — Gaze Tracking (Component 1)
Estimates whether you're looking at the camera vs. looking away, using
iris position relative to the eye corners. Logs the ratio of session time
spent looking away and prints a summary on quit.

Run: python gaze_tracker.py
Quit: press 'q'
"""

import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,   # required for iris landmarks (468-477)
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Landmark indices (MediaPipe Face Mesh, refined)
LEFT_EYE_CORNERS = (33, 133)     # outer, inner
RIGHT_EYE_CORNERS = (362, 263)   # inner, outer
LEFT_IRIS = 468
RIGHT_IRIS = 473

# How far the iris can drift from center (as a fraction of eye width)
# before we call it "looking away". Tune this after testing on yourself.
GAZE_AWAY_THRESHOLD = 0.35

looking_away_frames = 0
total_frames = 0
away_events = []   # (timestamp, duration) filled in on transition back to center
away_start_time = None

session_active = False
session_start = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    is_looking_away = False

    if session_active and result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        total_frames += 1

        def iris_offset_ratio(iris_idx, corner_a_idx, corner_b_idx):
            iris_x = lm[iris_idx].x * w
            ax = lm[corner_a_idx].x * w
            bx = lm[corner_b_idx].x * w
            eye_width = abs(bx - ax)
            if eye_width == 0:
                return 0
            eye_center = (ax + bx) / 2
            return (iris_x - eye_center) / eye_width

        left_ratio = iris_offset_ratio(LEFT_IRIS, *LEFT_EYE_CORNERS)
        right_ratio = iris_offset_ratio(RIGHT_IRIS, *RIGHT_EYE_CORNERS)
        avg_ratio = (left_ratio + right_ratio) / 2

        is_looking_away = abs(avg_ratio) > GAZE_AWAY_THRESHOLD

        # draw iris points for visual feedback
        for idx in (LEFT_IRIS, RIGHT_IRIS):
            cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        if is_looking_away:
            looking_away_frames += 1
            if away_start_time is None:
                away_start_time = time.time()
        else:
            if away_start_time is not None:
                duration = time.time() - away_start_time
                if duration > 0.5:   # ignore tiny blips
                    away_events.append((away_start_time - session_start, duration))
                away_start_time = None

    status_text = "LOOKING AWAY" if is_looking_away else "Eye contact OK"
    status_color = (0, 0, 255) if is_looking_away else (0, 200, 0)
    if session_active:
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    else:
        cv2.putText(frame, "Press 's' to start session", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if total_frames > 0:
        pct_away = 100 * looking_away_frames / total_frames
        cv2.putText(frame, f"Session away: {pct_away:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Interview Coach - Gaze Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        if not session_active:
            # starting fresh: reset all counters
            session_active = True
            session_start = time.time()
            looking_away_frames = 0
            total_frames = 0
            away_events = []
            away_start_time = None
        else:
            session_active = False
            if away_start_time is not None:
                duration = time.time() - away_start_time
                if duration > 0.5:
                    away_events.append((away_start_time - session_start, duration))
                away_start_time = None

cap.release()
cv2.destroyAllWindows()

# Close out any in-progress away event
if away_start_time is not None and session_start is not None:
    duration = time.time() - away_start_time
    if duration > 0.5:
        away_events.append((away_start_time - session_start, duration))

print("\n--- Session Summary ---")
if session_start is None:
    print("No session was started (press 's' to start next time).")
else:
    session_duration = time.time() - session_start
    print(f"Total session length: {session_duration:.1f}s")
    if total_frames > 0:
        print(f"Looking away: {100 * looking_away_frames / total_frames:.1f}% of frames")
    print(f"Number of away events: {len(away_events)}")
    for start_t, dur in away_events:
        print(f"  looked away at {start_t:.1f}s for {dur:.1f}s")
