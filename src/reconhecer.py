import cv2
import mediapipe as mp
import numpy as np

print("=== RECONHECIMENTO EM TEMPO REAL ===")
print("Pressione 'q' para sair")

mp_maos = mp.solutions.hands
hands = mp_maos.Hands()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)
    
    if resultado.multi_hand_landmarks:
        for landmarks in resultado.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp_maos.HAND_CONNECTIONS)
        cv2.putText(frame, "MAOS DETECTADAS", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Reconhecimento de Libras', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Fim!")