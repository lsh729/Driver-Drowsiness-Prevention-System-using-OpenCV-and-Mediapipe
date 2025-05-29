# detector/drowsiness_logic.py

import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

class DrowsinessEvaluator:
    def __init__(self):
        self.ear_threshold = 0.25
        self.mar_threshold = 0.7
        self.frame_count = 0
        self.drowsy_score = 0

    def compute_EAR(self, eye):
        # eye: 6 landmarks [(x1, y1), ..., (x6, y6)]
        A = euclidean(eye[0], eye[1])
        B = euclidean(eye[2], eye[3])
        C = euclidean(eye[4], eye[5])
        ear = (A + B) / (2.0 * C)
        return ear

    def compute_MAR(self, mouth):
        A = euclidean(mouth[0], mouth[1])
        B = euclidean(mouth[2], mouth[3])
        C = euclidean(mouth[4], mouth[5])
        D = euclidean(mouth[6], mouth[7])
        mar = (A + B+ C) / (3.0 * D)
        return mar

    def evaluate(self, landmarks):
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        mouth = landmarks['mouth']

        ear = (self.compute_EAR(left_eye) + self.compute_EAR(right_eye)) / 2.0
        mar = self.compute_MAR(mouth)

        state = "Normal"

        if ear < self.ear_threshold:
            self.frame_count += 1
            if self.frame_count >= 15:  # 눈을 15프레임 이상 감고 있으면 졸음
                self.drowsy_score += 1
                state = "Drowsy"
        else:
            self.frame_count = 0

        if mar > self.mar_threshold:
            self.drowsy_score += 1
            state = "Yawning"

        if self.drowsy_score >= 5:
            state = "Danger"

        return state
