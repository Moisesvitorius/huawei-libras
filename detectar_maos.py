# -*- coding: utf-8 -*-  
import cv2  
import mediapipe as mp  
import numpy as np  
  
print("Inicializando detector de maos para Libras...")  
  
mp_maos = mp.solutions.hands  
mp_desenho = mp.solutions.drawing_utils  
  
hands = mp_maos.Hands(  
    static_image_mode=False,  
    max_num_hands=2,  
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5  
)  
  
cap = cv2.VideoCapture(0)  
  
print("Pressione 'q' para sair...")  
while cap.isOpened():  
    sucesso, frame = cap.read()  
    if not sucesso:  
        print("Erro na webcam")  
        break  
  
    frame = cv2.flip(frame, 1)  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    resultado = hands.process(frame_rgb)  
  
    if resultado.multi_hand_landmarks:  
        for landmarks in resultado.multi_hand_landmarks:  
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)  
        print("Maos detectadas!")  
  
    cv2.imshow('HarmonyCare - Detecao de Libras', frame)  
  
