# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import mindspore as ms
import joblib
from collections import deque

print("="*60)
print("  HARMONYCARE - RECONHECIMENTO MINDSPORE")
print("="*60)

# Configurar MindSpore
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

# 1. CARREGAR MODELO
print("\n[INFO] Carregando modelo MindSpore...")

# Carregar checkpoint
class ModeloLibrasSimples(ms.nn.Cell):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = ms.nn.Conv1d(126, 64, kernel_size=3, padding=1, pad_mode='pad')
        self.bn1 = ms.nn.BatchNorm1d(64)
        self.relu = ms.nn.ReLU()
        self.pool = ms.nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = ms.nn.Conv1d(64, 128, kernel_size=3, padding=1, pad_mode='pad')
        self.bn2 = ms.nn.BatchNorm1d(128)
        
        self.flatten = ms.nn.Flatten()
        self.fc1 = ms.nn.Dense(128 * 15, 64)
        self.dropout = ms.nn.Dropout(keep_prob=0.5)
        self.fc2 = ms.nn.Dense(64, num_classes)
    
    def construct(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Criar modelo e carregar pesos
modelo = ModeloLibrasSimples()
param_dict = ms.load_checkpoint("models/modelo_libras_mindspore.ckpt")
ms.load_param_into_net(modelo, param_dict)
modelo.set_train(False)

# Carregar mapeamento
mapeamento = joblib.load("models/mapeamento_sinais.pkl")
id_para_sinal = mapeamento["id_para_sinal"]

print(f"[OK] Modelo carregado: reconhece {len(id_para_sinal)} sinais")

# 2. CONFIGURAR MEDIAPIPE
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
hands = mp_maos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 3. BUFFER PARA SEQUÊNCIA
buffer = deque(maxlen=30)
ultimo_sinal = None

# 4. FUNÇÃO PARA EXTRAIR LANDMARKS
def extrair_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)
    
    if resultado.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in resultado.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        
        if len(landmarks) < 126:
            landmarks.extend([0.0] * (126 - len(landmarks)))
        
        return landmarks[:126], resultado.multi_hand_landmarks
    return None, None

# 5. INICIALIZAR WEBCAM
print("\n[INFO] Inicializando webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERRO] Erro: Webcam nao encontrada!")
    exit()

print("[OK] Webcam pronta! Pressione 'q' para sair")

# Configurações da Interface
LARGURA_PAINEL = 300
AMARELO = (0, 255, 255)
VERDE = (0, 255, 0)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)

# 6. LOOP PRINCIPAL
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Extrair landmarks
    landmarks, hand_landmarks = extrair_landmarks(frame)
    
    # Desenhar landmarks se detectados
    if hand_landmarks:
        for landmarks_obj in hand_landmarks:
            mp_desenho.draw_landmarks(
                frame, landmarks_obj, mp_maos.HAND_CONNECTIONS)
    
    # Adicionar ao buffer
    if landmarks:
        buffer.append(landmarks)
    else:
        # Se não detectar mãos, limpar o buffer para não usar dados antigos
        if len(buffer) > 0:
            buffer.clear()
            sinal_atual = None
            ultimo_sinal = None
            confianca = 0
            
    # Reconhecer quando buffer estiver cheio
    sinal_atual = None
    confianca = 0
    
    if len(buffer) == 30:
        # Preparar dados
        sequencia = np.array(buffer, dtype=np.float32)
        sequencia = np.expand_dims(sequencia, axis=0)  # (1, 30, 126)
        
        # Converter para tensor MindSpore
        input_tensor = ms.Tensor(sequencia)
        
        # Predição
        output = modelo(input_tensor)
        output_np = output.asnumpy()
        
        # Softmax manual
        exp_output = np.exp(output_np - np.max(output_np))
        prob = exp_output / exp_output.sum(axis=1, keepdims=True)
        
        pred_id = np.argmax(prob[0])
        confianca = prob[0][pred_id] * 100
        
        if confianca > 20:  # Confiança mínima (Reduzido para 20%)
            sinal_atual = id_para_sinal.get(pred_id, "Desconhecido")
            ultimo_sinal = sinal_atual
    
    # --- CRIAÇÃO DA INTERFACE ---
    h, w, _ = frame.shape
    painel = np.zeros((h, LARGURA_PAINEL, 3), dtype=np.uint8)  # Painel preto
    
    # Título do Painel
    cv2.putText(painel, "HarmonyCare", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BRANCO, 2)
    cv2.line(painel, (20, 50), (LARGURA_PAINEL-20, 50), BRANCO, 1)

    # Status Buffer
    cor_bar = VERDE if len(buffer) == 30 else AMARELO
    cv2.putText(painel, f"Buffer: {len(buffer)}/30", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_bar, 1)
    
    # Barra de progresso do buffer
    comp_barra = int((len(buffer) / 30) * (LARGURA_PAINEL - 40))
    cv2.rectangle(painel, (20, 100), (20 + comp_barra, 110), cor_bar, -1)
    cv2.rectangle(painel, (20, 100), (LARGURA_PAINEL - 20, 110), BRANCO, 1)

    # Área de Detecção
    cv2.putText(painel, "SINAL DETECTADO:", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BRANCO, 1)
    
    texto_sinal = sinal_atual if sinal_atual else (ultimo_sinal if ultimo_sinal else "--")
    cor_sinal = VERDE if sinal_atual else (AMARELO if ultimo_sinal else BRANCO)
    
    # Centralizar ou quebrar texto se muito grande
    cv2.putText(painel, texto_sinal.upper(), (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_sinal, 3)

    # Confiança
    if confianca > 0:
        cv2.putText(painel, f"Confianca: {confianca:.1f}%", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BRANCO, 1)

    # Instruções Rodapé
    cv2.putText(painel, "Pressione 'q'", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(painel, "para sair", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Combinar Webcam + Painel
    interface_final = np.hstack((frame, painel))
    
    # Mostrar frame
    cv2.imshow('HarmonyCare - Reconhecimento MindSpore', interface_final)
    
    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# LIMPAR
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("Programa encerrado com sucesso!")
print("MindSpore funcionando perfeitamente!")
print("="*60)