# -*- coding: utf-8 -*-  
import cv2  
import mediapipe as mp  
  
print("Testando webcam e MediaPipe...")  
  
mp_maos = mp.solutions.hands  
hands = mp_maos.Hands()  
  
cap = cv2.VideoCapture(0)  
print("Webcam aberta! Pressione 'q' para sair")  
  
while True:  
    ret, frame = cap.read()  
    if not ret:  
        break  
  
    frame = cv2.flip(frame, 1)  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    resultado = hands.process(frame_rgb)  
  
    if resultado.multi_hand_landmarks:  
        for landmarks in resultado.multi_hand_landmarks:  
            mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)  
        print("Maos detectadas!")  
  
    cv2.imshow('Teste Webcam', frame)  
  
