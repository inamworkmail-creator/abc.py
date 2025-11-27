import streamlit as st
import cv2
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import tempfile
import os
import time

st.title("YOLO Detection with Correct Counting (IoU-based) + Voice")

# --- load model (update path if needed) ---
model = YOLO("C:/Users/User/Documents/all/sajid/best (1).pt")

# --- UI ---
run_cam = st.checkbox("Start Camera")

# --- session storage for counts & trackers ---
if "counts" not in st.session_state:
    st.session_state.counts = {}
if "tracked" not in st.session_state:
    # tracked: list of dicts {label, bbox:[x1,y1,x2,y2], last_seen:timestamp}
    st.session_state.tracked = []
if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = ""
# parameters
IOU_THRESHOLD = 0.4      # IoU above this -> same object
TRACK_TIMEOUT = 2.0      # seconds to keep a tracked object without updates

# --- helpers ---
def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0

def speak_label(label):
    # speak once per new counted object (simple)
    try:
        tts = gTTS(text=f"{label} detected", lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
            path = tf.name
        tts.save(path)
        playsound(path)
        os.remove(path)
    except Exception as e:
        # ignore audio errors to keep UI working
        print("TTS/play error:", e)

# placeholder for frame
frame_container = st.empty()

if run_cam:
    cam = cv2.VideoCapture(0)
    st.write("Camera running...")
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Camera not found.")
            break

        results = model(frame)[0]

        now = time.time()
        # clean old tracked entries
        st.session_state.tracked = [
            t for t in st.session_state.tracked if now - t["last_seen"] <= TRACK_TIMEOUT
        ]

        # process current detections
        for box in results.boxes:
            try:
                xy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
            except Exception:
                # fallback if different structure
                xy = [float(v) for v in box.xyxy[0]]
            cls = int(box.cls[0])
            label = model.names[cls]

            matched = False
            for t in st.session_state.tracked:
                if t["label"] == label and iou(xy, t["bbox"]) >= IOU_THRESHOLD:
                    # update existing tracked box timestamp and bbox
                    t["last_seen"] = now
                    t["bbox"] = xy
                    matched = True
                    break

            if not matched:
                # new object -> increment count and add to tracked list
                st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
                st.session_state.tracked.append({"label": label, "bbox": xy, "last_seen": now})
                # voice for this new count
                speak_label(label)

        # display camera frame (optional show boxes lightly)
        # draw tracked boxes for visual feedback
        for t in st.session_state.tracked:
            x1, y1, x2, y2 = map(int, t["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, t["label"], (x1, max(10, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_container.image(frame_rgb)

        # stop when checkbox turned off
        if not st.session_state.get("Start Camera", True):
            break

    cam.release()

# when camera stopped - show final counts
if not run_cam and st.session_state.counts:
    st.subheader("📌 Final Detected Items")
    total = sum(st.session_state.counts.values())
    st.write(f"**Total = {total}**")
    for item, cnt in st.session_state.counts.items():
        st.write(f"**{item} = {cnt}**")
