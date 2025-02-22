import cv2

class FaceDetector:
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml'):
        # Haar cascade file is available in OpenCV's data folder; you can also download it
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # Return the largest detected face as a bounding box
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            return faces[0]  # (x, y, w, h)
        return None