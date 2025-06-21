import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

focus_scores = []
timestamps = []

cap = cv2.VideoCapture(0)
start_time = time.time()

def eye_center(landmarks):
    x = sum([p.x for p in landmarks]) / len(landmarks)
    y = sum([p.y for p in landmarks]) / len(landmarks)
    return x, y

def distance(a, b, w, h):
    dx = (a[0] - b[0]) * w
    dy = (a[1] - b[1]) * h
    return np.sqrt(dx**2 + dy**2)

def calculate_focus(landmarks, w, h):
    left_iris = [landmarks[i] for i in LEFT_IRIS]
    right_iris = [landmarks[i] for i in RIGHT_IRIS]
    left_eye = [landmarks[i] for i in LEFT_EYE]
    right_eye = [landmarks[i] for i in RIGHT_EYE]

    li_center = eye_center(left_iris)
    le_center = eye_center(left_eye)
    ri_center = eye_center(right_iris)
    re_center = eye_center(right_eye)

    l_dist = distance(li_center, le_center, w, h)
    r_dist = distance(ri_center, re_center, w, h)

    scale = 25

    l_score = np.exp(-l_dist / scale)
    r_score = np.exp(-r_dist / scale)

    return (l_score + r_score) / 2

def smooth(scores):
    return sum(scores) / len(scores)

def moving_avg(data, window):
    return [sum(data[max(0, i - window + 1): i + 1]) / (i - max(0, i - window + 1) + 1) for i in range(len(data))]

last_logged_second = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    current_second = int(time.time() - start_time)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            score = calculate_focus(face_landmarks.landmark, w, h)
            focus_scores.append(score)

            if current_second > last_logged_second:
                if len(focus_scores) > 0:
                    average_focus = smooth(focus_scores)
                    timestamps.append((current_second, average_focus))
                    focus_scores.clear()
                last_logged_second = current_second

            cv2.putText(frame, f"Focus Score: {score:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        if current_second > last_logged_second:
            timestamps.append((current_second, 0.0))
            last_logged_second = current_second

    cv2.imshow("Focus Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV
df = pd.DataFrame(timestamps, columns=["Time", "Focus Score"])
df.to_csv("focus_score.csv", index=False)
