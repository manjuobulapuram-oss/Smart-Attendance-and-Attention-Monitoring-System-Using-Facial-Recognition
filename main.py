import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from deepface import DeepFace  # Optional for better emotion detection

# ====================== CONFIG ======================
KNOWN_FACES_DIR = "known_faces"  # Folder with student images named as "StudentName.jpg"
ATTENDANCE_FILE = "attendance.csv"
THRESHOLD_ATTENTION = 60  # % for low attention alert
EYE_CLOSED_THRESHOLD = 0.25  # EAR threshold
CONSEC_FRAMES_DROWSY = 15

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

print(f"Loaded {len(known_face_names)} known students.")

# Attendance tracking
attendance = {}
drowsy_counter = {}

def eye_aspect_ratio(eye):
    """Calculate EAR for one eye"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def calculate_attention(landmarks, frame_h, frame_w):
    """Simple attention score based on EAR, head pose approx"""
    if not landmarks:
        return 0
    
    # Left eye indices (MediaPipe Face Mesh)
    left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
    right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
    
    left_ear = eye_aspect_ratio(np.array([(p.x * frame_w, p.y * frame_h) for p in left_eye]))
    right_ear = eye_aspect_ratio(np.array([(p.x * frame_w, p.y * frame_h) for p in right_eye]))
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Rough head pose (using nose and forehead)
    attention_score = 100 if avg_ear > EYE_CLOSED_THRESHOLD else max(0, (avg_ear / EYE_CLOSED_THRESHOLD) * 100)
    return int(attention_score)

cap = cv2.VideoCapture(0)  # Webcam (0). For screen share, use other capture methods.

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Face recognition for attendance
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            if name not in attendance:
                attendance[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        face_names.append(name)
    
    # MediaPipe landmarks for attention
    results = face_mesh.process(rgb_frame)
    current_attentions = {}
    
    if results.multi_face_landmarks:
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape
            attention = calculate_attention(landmarks, h, w)
            
            # Optional: DeepFace emotion (slower, use sparingly)
            # try:
            #     emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            #     dominant = emotion[0]['dominant_emotion']
            # except: dominant = "neutral"
            
            x, y = face_locations[idx][3]*4, face_locations[idx][0]*4 if idx < len(face_locations) else (50, 50)
            cv2.putText(frame, f"{face_names[idx]}: {attention}%", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw rectangles
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Dashboard overlay
    avg_attention = np.mean(list(current_attentions.values())) if current_attentions else 0
    cv2.putText(frame, f"Class Avg Attention: {int(avg_attention)}%", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if avg_attention < THRESHOLD_ATTENTION:
        cv2.putText(frame, "LOW ATTENTION ALERT!", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow('AttenFace - Smart Attendance & Attention', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save attendance
df = pd.DataFrame(list(attendance.items()), columns=['Student', 'Timestamp'])
df.to_csv(ATTENDANCE_FILE, index=False)
print("Attendance saved!")