# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

print("=" * 60)
print("       INTELIMED - COLETOR DE LIBRAS")
print("=" * 60)

# Inicializar MediaPipe
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
hands = mp_maos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Criar pasta para dados
os.makedirs("data", exist_ok=True)

print("\n[INFO] Inicializando webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERRO] Erro: Webcam não encontrada!")
    exit()

print("[OK] Webcam pronta!")
print("\n[INFO] Instruções:")
print("1. Faça o sinal com suas mãos")
print("2. Pressione ESPAÇO para começar gravação")
print("3. Grave por 3 segundos (30 frames)")
print("4. Pressione ESC para cancelar")
print("=" * 60)

coletando = False
frames_coletados = []
frame_count = 0
sinal_atual = "dor"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Espelhar frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)
    
    # Desenhar landmarks se detectados
    if resultado.multi_hand_landmarks:
        for landmarks in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(
                frame, landmarks, mp_maos.HAND_CONNECTIONS)
    
    # Texto na tela
    if coletando:
        status = f"GRAVANDO: {frame_count}/30"
        cor = (0, 255, 0)  # Verde
    else:
        status = "PRESSIONE [ESPAÇO] PARA GRAVAR"
        cor = (0, 255, 255)  # Amarelo
    
    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
    cv2.putText(frame, f"Sinal: {sinal_atual}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "ESC = Sair | ESPAÇO = Gravar", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    
    # Mostrar frame
    cv2.imshow('InteliMed - Coletor de Libras', frame)
    
    # Teclas
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == 32:  # ESPAÇO
        if not coletando:
            print(f"\n[INFO] Gravando sinal: {sinal_atual}")
            coletando = True
            frames_coletados = []
            frame_count = 0
    
    # Coletar frames durante gravação
    if coletando and resultado.multi_hand_landmarks:
        # Extrair landmarks
        landmarks_frame = []
        for hand_landmarks in resultado.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks_frame.extend([lm.x, lm.y, lm.z])
        
        # Se tiver só uma mão, preencher com zeros
        if len(landmarks_frame) < 126:  # 21 pontos * 3 coord * 2 mãos
            landmarks_frame.extend([0.0] * (126 - len(landmarks_frame)))
        
        frames_coletados.append(landmarks_frame[:126])
        frame_count += 1
        
        # Progresso no terminal
        print(f"\r[INFO] Frames coletados: {frame_count}/30", end="")
        
        # Parar após 30 frames (~3 segundos)
        if frame_count >= 30:
            print(f"\n[OK] Sinal '{sinal_atual}' gravado com sucesso!")
            
            # Salvar dados
            dados = {
                "sinal": sinal_atual,
                "frames": frames_coletados,
                "num_frames": len(frames_coletados)
            }
            
            caminho_arquivo = os.path.join("data", f"{sinal_atual}.pkl")
            with open(caminho_arquivo, "wb") as f:
                pickle.dump(dados, f)
            
            print(f"[OK] Salvo em: {caminho_arquivo}")
            
            # Próximo sinal
            sinais = ["dor"]
            idx = sinais.index(sinal_atual) if sinal_atual in sinais else 0
            if idx + 1 < len(sinais):
                sinal_atual = sinais[idx + 1]
                print(f"\n>> Próximo sinal: {sinal_atual}")
                print("Pressione ESPAÇO para gravar próximo sinal")
            else:
                print("\n[OK] Todos os sinais gravados!")
                print("Pressione ESC para sair")
            
            coletando = False

# Limpar
cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 60)
print("[OK] Coleta finalizada!")
print("[INFO] Dados salvos na pasta: data/")
print("=" * 60)
