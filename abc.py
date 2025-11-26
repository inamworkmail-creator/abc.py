import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import os
import time

st.set_page_config(page_title="Live YOLO Detection with Voice", layout="wide")

# ---------------------------
# Load YOLO model
# ---------------------------
MODEL_PATH = "C:/Users/User/Documents/all/sajid/best (1).pt"   # your trained model
model = YOLO(MODEL_PATH)

# Flag to avoid repeated speaking
last_spoken = ""
speak_delay = 3   # seconds
last_speak_time = 0

# ---------------------------
# Function: Speak detected object
# ---------------------------
def speak_object(label):
    global last_spoken, last_speak_time
    
    # avoid repeated speech
    if label == last_spoken and (time.time() - last_speak_time < speak_delay):
        return
    
    last_spoken = label
    last_speak_time = time.time()

    tts = gTTS(text=f"Prohibited item detected: {label}", lang="en")

    # Create safe temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        audio_path = tmp_file.name

    tts.save(audio_path)
    playsound(audio_path)
    os.remove(audio_path)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔴 Live Object Detection with Voice Alert (YOLO)")

run_cam = st.checkbox("Start Camera")

frame_window = st.image([])

# ---------------------------
# Camera Loop
# ---------------------------
if run_cam:
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            st.warning("Camera not found.")
            break

        # YOLO Detection
        results = model(frame)[0]

        # Draw boxes
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            label = model.names[cls]

            # speak detected item
            speak_object(label)

            # draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{label}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cam.release()
