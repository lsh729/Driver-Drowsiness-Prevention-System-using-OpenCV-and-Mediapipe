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
            if self.frame_count % 30 == 0:  # 눈 감은지 1초마다 2점
                self.drowsy_score += 2
                state = "Drowsy"
        else:
            self.frame_count = 0

       # 하품 감지: 1초 이상 유지 시 1번만 5점 추가
        if mar > self.mar_threshold:
            self.yawn_frame_count += 1
            if self.yawn_frame_count >= 30 and not self.has_yawned:
                self.drowsy_score += 5
                self.has_yawned = True
                state = "Yawning"
        else:
            self.yawn_frame_count = 0
            self.has_yawned = False

        # 졸음 점수 기준 상태
        if self.drowsy_score >= 30:
            state = "Danger"
        elif self.drowsy_score >= 20:
            state = "Warning"

        # 점수 감소: 눈 뜨고 입 닫았을 때만 서서히 감소
        if ear >= self.ear_threshold and mar <= self.mar_threshold:
            if self.drowsy_score > 0:
                self.drowsy_score -= 0.05

        return state