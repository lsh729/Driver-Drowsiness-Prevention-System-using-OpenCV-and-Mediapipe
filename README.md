Driver Drowsiness Prevention System using OpenCV and Mediapipe


## 설명

OpenCV와 Mediapipe를 이용하여 눈 감김, 하품, 고개 숙임을 탐지하여 운전자의 졸음 여부를 판단하고 위험 수준에 따라 경보음이나 음악을 재생하여 사고를 예방하는 데에 도움을 주는 Python 기반 실시간 졸음 감지 프로그램입니다. Mediapipe를 활용해 얼굴의 landmark들중 졸음 운전 판단에 필요한 부분들을 추출하여 OpenCV로 영상을 입력하고 landmark 좌표들을 시각화하며, 위험 수준을 3단계로 분류한 뒤, 각기 다른 색상으로 표시하여 더욱 가시적으로 인지할 수 있도록 구현하였습니다.




## 필수 설치 패키지

pip install opencv-python  
pip install mediapipe  
pip install pygame  
numpy  


Python 버전은 3.8 ~ 3.12를 권장합니다.





## 폴더 구조

.  
├── main.py  
├── detector/  
│   ├── drowsiness_logic.py  
│   └── face_detector.py  
├── songs/  
│   ├── alarm.mp3  
│   ├── wake1.mp3  
│   └── ...  
├── demo/  
│   ├── 시연영상.mkv  
│   └── thumbnail.png  
├── README.md  





## 주요 기능


-   눈 감김(EAR) 감지 : Mediapipe로부터 눈 좌표를 추출하고 EAR(Eye Aspect Ratio)를 계산하여 일정 시간 이상 감긴 상태를 탐지함.


-    하품(MAR) 감지 : 입 좌표 기반 MAR(Mouth Aspect Ratio)를 통해 하품 여부를 감지. 15프레임 이상 입이 열려 있으면 하품으로 판단하고 점수 반영  



-    고개 숙임(Pitch) 감지 : 3D-2D 투영(PnP) 방식으로 얼굴 기울기(특히 pitch angle)를 계산하여 고개를 숙인 상태인지 판단  


-    졸음 점수 기반 상태 평가 : 눈, 입, 고개 데이터를 종합하여 졸음 점수를 계산. 점수에 따라 세 가지 상태로 분류: Normal, Warning, Danger  


-    단계별 경고음 재생 : 상태에 따라 다른 오디오 반응: Danger 상태는 경고음(alarm.mp3), Warning 상태는 랜덤 음악(wake1~3.mp3), Normal 상태는 무음  


-    시각적 효과 : 화면에 졸음 점수와 state를 출력, Danger 상태일 경우 화면에 빨간 테두리가 깜빡거려  시각적인 위험성을 인지시킴.  






## 간략한 코드 설명



1) face_detector.py 코드 설명

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

- OpenCV는 BGR이지만 Mediapipe는 RGB이므로 변환


        self.LEFT_EYE_IDX = [160, 144, 158, 153, 33, 133]
        self.RIGHT_EYE_IDX = [385, 380, 387, 373, 362, 263]
        self.MOUTH_IDX = [13, 14, 80, 402, 271, 88, 61, 291]
        self.HEADPOSE_IDX = [1, 33, 61, 199, 263, 291]

- eye, mouth, heapose의 좌표



          def extract_points(index_list):
                  return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in index_list]
  
              return {
                  'left_eye': extract_points(self.LEFT_EYE_IDX),
                  'right_eye': extract_points(self.RIGHT_EYE_IDX),
                  'mouth': extract_points(self.MOUTH_IDX),
                  'headpose': extract_points(self.HEADPOSE_IDX)

- Mediapipe는 0~1 사이 정규화된 좌표를 반환하므로, eye, mouth, heapose의 좌표들을 OpenCV 프레임 크기의 픽셀 좌표(x,y) 로 변환하여 EAR, MAR, 고개 각도 계산에 사용.




2) main.py 코드 설명

        cv2.VideoCapture(0)
        (cap.read())

- 웹캠을 열고 매 프레임마다 캠에서 이미지를 받기.



        landmarks = face.get_landmarks(frame)

- 얼굴이 인식되면 Mediapipe 기반 get_landmarks()를 통해 eye, mouth, headpose 좌표를 추출.




         score = evaluator.drowsy_score
            if score < 20:
                color = (0, 255, 0)
            elif score < 30:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            if state == "Danger" and (frame_count // 10) % 2 == 0:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            
- 점수에 따라 색상을 변경( 20점 미만 초록, 20~29점은 노랑, 30점 이상 빨강), "Danger" state일 경우, 화면에 빨간색 테두리가 깜빡거림.



3) drowsiness_logic.py 코드 설명

        self.ear_threshold = 0.23
        self.mar_threshold = 0.65

- ear, mar 기준값



  pygame.mixer.init()
        self.current_audio = "none"
        self.wake_up_songs = [
          "songs/wake1.mp3",
          "songs/wake2.mp3",
          "songs/wake3.mp3",
          ]

- pygame을 이용한 오디오 재생 초기화, 음악 재생 목록 준비




        def compute_EAR(self, eye):
            A = euclidean(eye[0], eye[1])
            B = euclidean(eye[2], eye[3])
            C = euclidean(eye[4], eye[5])
            ear = (A + B) / (2.0 * C)
            return ear

- 좌표 6개로 EAR 계산 함수, 눈이 감길수록 EAR 값은 감소 (기준값 0.23)




                if ear < self.ear_threshold:
                    self.eye_close_counter += 1
                    self.eye_open_frame = 0


- EAR 값이 0.23보다 작을 경우, 눈 감은 상태로 판단




            if self.eye_close_counter >= 120:
                self.drowsy_score = 30        


- 눈을 120프레임(약 4초) 이상 감은 상태일 경우, 강제 "Danger" 상태 진입 (점수 즉시 30)



            if self.eye_close_counter >= 30 and self.eye_close_counter % 15 == 0:
                self.drowsy_score += 2


- 30프레임(약 1초) 이상 감은 상태일 경우, 15프레임마다 점수 +2



              else:
                  if 1 < self.eye_close_counter <= 3:
                    print("Blink detected")

- 2~3프레임 동안 감긴 상태였을 경우엔 눈 깜빡임(Blink)로 인식 (점수에 반여하지 않음)
    -> 빠르게 깜빡일 때 점수가 올라가는 경우 방지



              self.eye_close_counter = 0
              self.eye_open_frame += 1

- 눈 뜬 상태로 인식하므로 eye_open_frame 누적



              if self.eye_open_frame >= 15 and self.eye_open_frame % 15 == 0:
                  self.drowsy_score = max(0, self.drowsy_score - 1)

- 눈을 15프레임 이상 뜨고 있는 동안 점수를 1씩 감소 (최하 점수 0)



        def compute_MAR(self, mouth):
            A = euclidean(mouth[0], mouth[1])
            B = euclidean(mouth[2], mouth[3])
            C = euclidean(mouth[4], mouth[5])
            D = euclidean(mouth[6], mouth[7])
            mar = (A + B + C) / (3.0 * D)
        return mar

- 좌표 8개로 MAR 계산 함수, 입을 벌릴 수록 MAR값 증가 (기준값 0.65)



            if mar > self.mar_threshold:
                self.yawn_frame_count += 1

- MAR가 0.65 이상이면 입이 열린 상태



            if self.yawn_frame_count >= 15 and not self.has_yawned:
                self.drowsy_score += 5
                self.has_yawned = True
                state = "Yawning"
        else:
            self.yawn_frame_count = 0
            self.has_yawned = False


15프레임 이상 입이 열린 상태이고 처음 감지된 상태인 경우, yawning 상태로 점수 5점 증가, 중복 방지



          def compute_pitch_angle(self, image_points, frame_shape):

            model_points = np.array([
              (0.0, 0.0, 0.0),        # 코 끝 (기준점)
              (-30.0, -30.0, -30.0),  # 왼쪽 눈
              (-30.0, 30.0, -30.0),   # 왼쪽 입
              (0.0, 60.0, -50.0),     # 턱 (아래쪽)
              (30.0, -30.0, -30.0),   # 오른쪽 눈
              (30.0, 30.0, -30.0)     # 오른쪽 입
            ])


- Mediapipe의 얼굴 좌표 6개




          height, width = frame_shape[:2]
                focal_length = width
                center = (width / 2, height / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")
                dist_coeffs = np.zeros((4, 1))
  
          success, rotation_vector, translation_vector = cv2.solvePnP(
              model_points,
              np.array(image_points, dtype="double"),
              camera_matrix,
              dist_coeffs
          )
  
          rmat, _ = cv2.Rodrigues(rotation_vector)
          pitch = np.arcsin(-rmat[2][1]) * 180.0 / np.pi  
          return pitch


- 3D 영상과 2D 좌표를 매칭할 수 있게 camera matrix를 설정, slovepnp로 회전 벡터 추정 후 pitch 값 계산


          if pitch > 27:
                      self.head_down_counter += 1
                      if self.head_down_counter >= 5:
                          self.drowsy_score = 30
                          state = "Danger"
                  else:
                      self.head_down_counter = 0



- pitch가 27이상 5프레임 이상 감지되었을 경우, 점수 30점으로 즉시 "Danger" 상태 돌입 



          if self.drowsy_score >= 30:
                      state = "Danger"
                      if self.current_audio != "alarm":
                          pygame.mixer.music.stop()
                          pygame.mixer.music.load("songs/alarm.mp3")
                          pygame.mixer.music.play(-1)  
                          self.current_audio = "alarm"


- 졸음 점수가 30 이상이면 Danger 상태, 재생중인 오디오가 alarm이 아닐 경우, 기존 음악 중지후 alarm 무한 반복(-1)


            elif 20 <= self.drowsy_score < 30:
                state = "Warning"

                if self.current_audio == "alarm":
                    pygame.mixer.music.stop()
                    self.current_audio = "none"

            if not pygame.mixer.music.get_busy():
                next_song = random.choice(self.wake_up_songs)
                while next_song == self.current_audio:
                    next_song = random.choice(self.wake_up_songs)

                pygame.mixer.music.load(next_song)
                pygame.mixer.music.play()
                self.current_audio = next_song

- 졸음 점수가 20점과 29점사이면 Warning 상태, alarm.mp3가 재생 중이면 중지 or 음악이 재생되고 있지 않으면 wake1,2,3중 하나 랜덤 재생, 계속 반복

                else:
                    state = "Normal"
                    if self.current_audio != "none":
                        pygame.mixer.music.stop()
                        self.current_audio = "none"


- 졸음 점수가 20점 미만이면 Normal 상태, 재생 중인 음악이 있을 경우 중지




## 시연 영상



[![Demo Video](demo/thumbnail.png)](demo/demo.mkv)




## 참고 자료 링크

Eye Aspect Ratio (EAR)
논문: Real-Time Eye Blink Detection using Facial Landmarks

(https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
(Soukupová & Čech, 2016)



Head Pose Estimation

(https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)


Mediapipe Face Mesh

(https://google.github.io/mediapipe/solutions/face_mesh.html)

