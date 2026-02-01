import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

def extract_gesture_landmarks(video_path, max_frames=64):
    cap = cv2.VideoCapture(video_path)
    all_frames = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:

        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)

            frame_landmarks = []

            # Pose
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0] * POSE_LANDMARKS * 3)

            # Left Hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0] * HAND_LANDMARKS * 3)

            # Right Hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0] * HAND_LANDMARKS * 3)

            all_frames.append(frame_landmarks)
            frame_count += 1

    cap.release()
    return np.array(all_frames)  # shape: (T, N*3)
