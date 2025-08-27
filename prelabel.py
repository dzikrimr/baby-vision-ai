# prelabel.py
import cv2
import numpy as np
import pandas as pd
import logging
import psutil 
from src.detector import SleepDetector, mouth_aspect_ratio_mp
from src.utils import eye_aspect_ratio

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

def extract_features_and_label(frame, detector, prev_landmarks=None):
    # Resize frame
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = True

    face_results = detector.face_mesh.process(rgb_frame)
    pose_results = detector.pose.process(rgb_frame)

    features = []
    labels = [0, 0, 0, 0, 0]  # [eye, mouth, movement, pacifier, rollover]
    current_landmarks = None

    # Mata & Mulut 
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        right_eye_indices = [33, 160, 158, 133, 153, 144]

        left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices])
        right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        try:
            mar = mouth_aspect_ratio_mp(None, landmarks, h, w)
        except Exception as e:
            print(f"Error calculating MAR: {e}")
            mar = 0.0

        features.extend([ear, mar])
        labels[0] = 1 if ear > 0.20 else 0 if ear < 0.15 else 2  
        labels[1] = 1 if mar > 0.30 else 0
    else:
        features.extend([0.0, 0.0])
        labels[0] = 0
        labels[1] = 0

    # Gerakan
    movement = 0.0
    if pose_results and pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        selected_landmarks = [
            detector.mp_pose.PoseLandmark.NOSE,
            detector.mp_pose.PoseLandmark.LEFT_SHOULDER,
            detector.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            detector.mp_pose.PoseLandmark.LEFT_WRIST,
            detector.mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        current_landmarks = {
            idx: (landmarks[idx.value].x, landmarks[idx.value].y, landmarks[idx.value].z)
            for idx in selected_landmarks
        }

        if prev_landmarks:
            total_distance = sum(
                np.sqrt(sum((c - p) ** 2 for c, p in zip(current_landmarks[idx], prev_landmarks[idx])))
                for idx in current_landmarks
            )
            movement = total_distance / len(current_landmarks)

        # update prev_landmarks setiap frame
        prev_landmarks = current_landmarks.copy()

        # threshold lebih realistis
        labels[2] = 1 if movement > 0.003 else 0  

    features.append(movement)

    # Empeng
    try:
        pacifier = 1.0 if "Dipakai" in detector.detect_pacifier(frame) else 0.0
    except Exception as e:
        print(f"Error detecting pacifier: {e}")
        pacifier = 0.0
    features.append(pacifier)
    labels[3] = pacifier

    # Rollover
    if pose_results and pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        nose_y = landmarks[detector.mp_pose.PoseLandmark.NOSE.value].y
        avg_body_y = (landmarks[detector.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                      landmarks[detector.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y +
                      landmarks[detector.mp_pose.PoseLandmark.LEFT_HIP.value].y +
                      landmarks[detector.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 4
        rollover = 1.0 if nose_y > avg_body_y + 0.05 or not face_results.multi_face_landmarks else 0.0
        labels[4] = rollover
    else:
        rollover = 1.0 if not face_results.multi_face_landmarks else 0.0
        labels[4] = rollover
    features.append(rollover)

    return np.array(features), labels, prev_landmarks

# Main Loop 
dataset = ["dataset/babyincrib_set.mp4"]
csv_file = "dataset/annotations.csv"

# Initialize CSV
pd.DataFrame(columns=["video_path", "frame_id", "time_sec", "eye_status", "mouth_status",
                      "movement_status", "pacifier_status", "rollover_status"]).to_csv(csv_file, index=False)

detector = SleepDetector()

for video_path in dataset:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} with FPS: {fps}, Total Frames: {total_frames}, Duration: {total_frames/fps:.2f}s")

    prev_landmarks = None
    frame_id = 0
    skip_frames = 1
    batch_size = 500
    batch_annotations = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_frames == 0:
            try:
                cpu_usage = psutil.cpu_percent()
                mem_usage = psutil.virtual_memory().percent
                print(f"Frame {frame_id}: CPU={cpu_usage}%, Memory={mem_usage}%")

                features, labels, prev_landmarks = extract_features_and_label(frame, detector, prev_landmarks)

                # Debug movement
                print(f"Frame {frame_id}: movement={features[2]:.6f}, label={labels[2]}")

                batch_annotations.append({
                    "video_path": video_path,
                    "frame_id": frame_id,
                    "time_sec": frame_id / fps,
                    "eye_status": labels[0],
                    "mouth_status": labels[1],
                    "movement_status": labels[2],
                    "pacifier_status": labels[3],
                    "rollover_status": labels[4]
                })

                if len(batch_annotations) >= batch_size:
                    pd.DataFrame(batch_annotations).to_csv(csv_file, mode='a', header=False, index=False)
                    batch_annotations = []
            except Exception as e:
                print(f"Error processing frame {frame_id} in {video_path}: {e}")

        frame_id += 1

    if batch_annotations:
        pd.DataFrame(batch_annotations).to_csv(csv_file, mode='a', header=False, index=False)

    cap.release()
    print(f"Finished processing {video_path}")
