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
        
        # 점수 따라 색상 변경
        score = evaluator.drowsy_score
        if score < 20:
            color = (0, 255, 0)
        elif score < 30:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
            
        # 점수와 상태 출력    
        cv2.putText(frame, f"Score: {score}", (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.putText(frame, f"State: {state}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 눈/입 좌표 시각화
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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
else:
    print("✅ 카메라 열림 성공")