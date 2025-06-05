# drowsiness_logic.py

import numpy as np
import pygame
import random
import cv2

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

class DrowsinessEvaluator:
    def __init__(self):
        self.ear_threshold = 0.23
        self.mar_threshold = 0.65
        self.frame_count = 0
        self.drowsy_score = 0
        self.yawn_frame_count = 0
        self.has_yawned = False         
        self.eye_close_counter = 0
        self.eye_open_frame = 0
        self.head_down_counter = 0
        self.head_down_threshold = 100
        
        pygame.mixer.init()
        self.current_audio = "none"
        self.wake_up_songs = [
            "songs/wake1.mp3",
            "songs/wake2.mp3",
            "songs/wake3.mp3",
            ]

    def compute_EAR(self, eye):
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
        mar = (A + B + C) / (3.0 * D)
        return mar

    def compute_pitch_angle(self, image_points, frame_shape):
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (-30.0, -30.0, -30.0),  # Left eye
            (-30.0, 30.0, -30.0),   # Left mouth
            (0.0, 60.0, -50.0),     # Chin
            (30.0, -30.0, -30.0),   # Right eye
            (30.0, 30.0, -30.0)     # Right mouth
        ])

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

    def evaluate(self, landmarks, frame):
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        mouth = landmarks['mouth']
        headpose_pts = landmarks['headpose']

        ear = (self.compute_EAR(left_eye) + self.compute_EAR(right_eye)) / 2.0
        mar = self.compute_MAR(mouth)
        pitch = self.compute_pitch_angle(headpose_pts, frame.shape)

        
        state = "Normal"
        print(f"[DEBUG] pitch angle: {pitch:.2f} deg")
        
        if ear < self.ear_threshold:
            self.eye_close_counter += 1
            self.eye_open_frame = 0

            if self.eye_close_counter >= 120:
                self.drowsy_score = 30        

            if self.eye_close_counter >= 30 and self.eye_close_counter % 15 == 0:
                self.drowsy_score += 2

        else:
            if 1 < self.eye_close_counter <= 3:
                print("Blink detected")


            self.eye_close_counter = 0
            self.eye_open_frame += 1

            if self.eye_open_frame >= 15 and self.eye_open_frame % 15 == 0:
                self.drowsy_score = max(0, self.drowsy_score - 1)
        
            
        if mar > self.mar_threshold:
            self.yawn_frame_count += 1
            if self.yawn_frame_count >= 15 and not self.has_yawned:
                self.drowsy_score += 5
                self.has_yawned = True
                state = "Yawning"
        else:
            self.yawn_frame_count = 0
            self.has_yawned = False

        if pitch > 27:
            self.head_down_counter += 1
            if self.head_down_counter >= 5:
                self.drowsy_score = 30
                state = "Danger"
        else:
            self.head_down_counter = 0

        if self.drowsy_score >= 30:
            state = "Danger"
            if self.current_audio != "alarm":
                pygame.mixer.music.stop()
                pygame.mixer.music.load("songs/alarm.mp3")
                pygame.mixer.music.play(-1)  
                self.current_audio = "alarm"

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
        else:
            state = "Normal"
            if self.current_audio != "none":
                pygame.mixer.music.stop()
                self.current_audio = "none"

        return state