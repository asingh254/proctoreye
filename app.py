import cv2
import streamlit as st
from modules.face_detector import FaceDetector
from modules.gaze_tracker import GazeTracker
from modules.alert_system import AlertSystem

# Initialize modules
face_detector = FaceDetector()
gaze_tracker = GazeTracker(device='cpu')
alert_system = AlertSystem()

st.title("ProctorEye: AI-Powered Anti-Cheating Solution")
st.write("Monitoring gaze and head movements during exams")

if 'stop' not in st.session_state:
    st.session_state.stop = False

# Create stop button once
if st.button('Stop', key="stop_button"):
    st.session_state.stop = True

cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])
alert_box = st.empty()

while cap.isOpened() and not st.session_state.stop:
    ret, frame = cap.read()
    if not ret:
        break

    face_box = face_detector.detect_face(frame)
    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        landmarks = gaze_tracker.get_landmarks(frame, face_box)
        if landmarks is not None:
            direction = gaze_tracker.get_eye_direction(frame, landmarks)
            cv2.putText(frame, f"Eye Direction: {direction}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if direction == "Looking Away":
                alert_message = alert_system.log_alert("Suspicious behavior detected!")
                alert_box.warning(alert_message)
    else:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
cv2.destroyAllWindows()
