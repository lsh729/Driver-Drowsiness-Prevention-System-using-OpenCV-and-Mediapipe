import numpy as np
import pygame
import random

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
        
        pygame.mixer.init()
        self.current_audio = "none"
        self.wake_up_songs = [
            "songs/wake1.mp3",
            "songs/wake2.mp3",
            "songs/wake3.mp3",
            "songs/wake4.mp3",
            "songs/wake5.mp3"
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

    def evaluate(self, landmarks):
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        mouth = landmarks['mouth']

        ear = (self.compute_EAR(left_eye) + self.compute_EAR(right_eye)) / 2.0
        mar = self.compute_MAR(mouth)
        
        state = "Normal"

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

        if self.drowsy_score >= 30:
            state = "Danger"
            if self.current_audio != "alarm":
                pygame.mixer.music.stop()
                pygame.mixer.music.load("alarm.mp3")
                pygame.mixer.music.play(-1)  
                self.current_audio = "alarm"

        elif 20 <= self.drowsy_score < 30:
            state = "Warning"
            if self.current_audio != "music":
                self.current_audio = "music"
            if not pygame.mixer.music.get_busy():
                next_song = random.choice(self.wake_up_songs)
                pygame.mixer.music.load(next_song)
                pygame.mixer.music.play()

        else:
            state = "Normal"
            if self.current_audio != "none":
                pygame.mixer.music.stop()
                self.current_audio = "none"

        return state
