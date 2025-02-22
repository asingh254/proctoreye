import cv2
import numpy as np
import face_alignment

class GazeTracker:
    def __init__(self, device='cpu'):
        # Initialize face-alignment with 2D landmarks
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

    def get_landmarks(self, frame, face_box):
        """
        Uses face-alignment to detect facial landmarks.
        The face_box can be used to crop the face for faster detection.
        """
        x, y, w, h = face_box
        face_img = frame[y:y+h, x:x+w]
        preds = self.fa.get_landmarks(face_img)
        if preds is not None:
            # Adjust landmarks back to the coordinates of the full frame
            landmarks = preds[0]
            landmarks[:, 0] += x
            landmarks[:, 1] += y
            return landmarks
        return None

    def get_eye_direction(self, frame, landmarks):
        """
        Determines if the user is looking forward or away using landmarks.
        Uses landmarks for left (36-41) and right (42-47) eyes (if using a 68-landmark model).
        For face-alignment, the order is the same as in dlib.
        """
        # Ensure we have 68 landmarks
        if landmarks.shape[0] != 68:
            return "Unknown"
        
        # Left eye landmarks (indices 36-41) and right eye (42-47)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        def eye_center(eye):
            return np.mean(eye, axis=0).astype("int")
        
        left_center = eye_center(left_eye)
        right_center = eye_center(right_eye)
        eye_center_avg = ((left_center[0] + right_center[0]) // 2,
                          (left_center[1] + right_center[1]) // 2)

        frame_center = (frame.shape[1]//2, frame.shape[0]//2)
        dx = abs(eye_center_avg[0] - frame_center[0])
        dy = abs(eye_center_avg[1] - frame_center[1])
        
        if dx > 50 or dy > 50:
            return "Looking Away"
        return "Looking Forward"
