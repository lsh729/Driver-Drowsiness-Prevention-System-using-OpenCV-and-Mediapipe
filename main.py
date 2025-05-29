# main.py

import cv2
from detector.face_detector import FaceLandmarkDetector
from detector.drowsiness_logic import DrowsinessEvaluator

# 객체 초기화
cap = cv2.VideoCapture(0)
face = FaceLandmarkDetector()
evaluator = DrowsinessEvaluator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 랜드마크 추출
    landmarks = face.get_landmarks(frame)

    if landmarks:
        state = evaluator.evaluate(landmarks)

        # 화면에 상태 출력
        cv2.putText(frame, f"State: {state}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 눈/입 좌표 디버깅 시 시각화 (선택)
        for pt in landmarks['left_eye'] + landmarks['right_eye'] + landmarks['mouth']:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

    else:
        cv2.putText(frame, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 화면 출력
    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
