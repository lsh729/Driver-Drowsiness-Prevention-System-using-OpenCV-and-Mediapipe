# detector/face_detector.py

import mediapipe as mp
import cv2

class FaceLandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

        # 왼쪽/오른쪽 눈, 입 좌표 인덱스 (Mediapipe 기준)
        self.LEFT_EYE_IDX = [160, 144, 158, 153, 33, 133]
        self.RIGHT_EYE_IDX = [385, 380, 387, 373, 362, 263]
        self.MOUTH_IDX = [13, 14, 80, 402, 271, 88, 61, 291]
        self.HEADPOSE_IDX = [1, 33, 61, 199, 263, 291]

    def get_landmarks(self, frame):
        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            # 추출된 좌표를 (x, y) 픽셀 단위로 변환
            def extract_points(index_list):
                return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in index_list]

            return {
                'left_eye': extract_points(self.LEFT_EYE_IDX),
                'right_eye': extract_points(self.RIGHT_EYE_IDX),
                'mouth': extract_points(self.MOUTH_IDX),
                'headpose': extract_points(self.HEADPOSE_IDX)
            }

        return None
