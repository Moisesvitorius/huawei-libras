# -*- coding: utf-8 -*-
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import pickle
import os
from collections import defaultdict

print("="*60)
print("       HARMONYCARE - TREINAMENTO DO MODELO")
print("="*60)

# 1. CARREGAR DADOS
print("\n📂 Carregando dados coletados...")
dados = []
rotulos = []

pasta_dados = "data"
if not os.path.exists(pasta_dados):
    print("❌ Erro: Pasta 'data' não encontrada!")
    print("   Execute primeiro: python src\\coletar_dados.py")
    exit()

for arquivo in os.listdir(pasta_dados):
    if arquivo.endswith(".pkl"):
        caminho = os.path.join(pasta_dados, arquivo)
        with open(caminho, "rb") as f:
            dados_sinal = pickle.load(f)
        
        sinal = dados_sinal["sinal"]
        frames = dados_sinal["frames"]
        
        # Garantir que todos tenham 30 frames (padding/truncamento)
        seq_len = 30
        if len(frames) > seq_len:
            # Amostrar uniformemente
            indices = np.linspace(0, len(frames)-1, seq_len, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < seq_len:
            # Repetir último frame
            ultimo_frame = frames[-1] if frames else [0]*126
            frames = frames + [ultimo_frame] * (seq_len - len(frames))
        
        dados.append(np.array(frames, dtype=np.float32))
        rotulos.append(sinal)

print(f"✅ Dados carregados: {len(dados)} amostras")

# Mapeamento sinais → números
sinais_unicos = list(set(rotulos))
sinal_para_id = {sinal: i for i, sinal in enumerate(sinais_unicos)}
id_para_sinal = {i: sinal for i, sinal in enumerate(sinais_unicos)}

print(f"📊 Sinais: {sinais_unicos}")
print(f"🔢 Mapeamento: {sinal_para_id}")

# Converter rótulos para números
rotulos_numericos = [sinal_para_id[r] for r in rotulos]

# 2. CRIAR MODELO
print("\n🧠 Criando modelo LSTM para Libras...")

class ModeloLibras(nn.Cell):
    def __init__(self, num_sinais=5):
        super().__init__()
        # LSTM: 126 features → 128 hidden units
        self.lstm = nn.LSTM(
            input_size=126,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc1 = nn.Dense(256, 64)  # 128*2 (bidirecional)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.fc2 = nn.Dense(64, num_sinais)
    
    def construct(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x)
        ultimo = lstm_out[:, -1, :]  # Último time step
        x = self.fc1(ultimo)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# 3. PREPARAR DATASET
print("\n📊 Preparando dataset...")
X = np.array(dados)  # Shape: (n_amostras, 30, 126)
y = np.array(rotulos_numericos, dtype=np.int32)

# Embaralhar
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Dividir treino/teste
split = int(0.8 * len(X))
X_treino, X_teste = X[:split], X[split:]
y_treino, y_teste = y[:split], y[split:]

print(f"   Treino: {len(X_treino)} amostras")
print(f"   Teste: {len(X_teste)} amostras")

# 4. TREINAR
print("\n🚀 Iniciando treinamento...")
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

modelo = ModeloLibras(num_sinais=len(sinais_unicos))

# Loss e otimizador
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(modelo.trainable_params(), learning_rate=0.001)

# Loop de treino simples
modelo.set_train()
num_epocas = 20
batch_size = 4

for epoca in range(num_epocas):
    perda_total = 0
    acertos = 0
    total = 0
    
    for i in range(0, len(X_treino), batch_size):
        batch_X = X_treino[i:i+batch_size]
        batch_y = y_treino[i:i+batch_size]
        
        if len(batch_X) == 0:
            continue
        
        # Converter para tensor MindSpore
        X_tensor = ms.Tensor(batch_X)
        y_tensor = ms.Tensor(batch_y, dtype=ms.int32)
        
        # Forward
        output = modelo(X_tensor)
        loss = loss_fn(output, y_tensor)
        
        # Backward (simplificado)
        grad_fn = ms.value_and_grad(modelo, None, optimizer.parameters)
        (loss_value, grads) = grad_fn(X_tensor, y_tensor)
        optimizer(grads)
        
        perda_total += loss_value.asnumpy()
        
        # Calcular acurácia
        preds = np.argmax(output.asnumpy(), axis=1)
        acertos += np.sum(preds == batch_y)
        total += len(batch_y)
    
    acuracia = acertos / total if total > 0 else 0
    
    print(f"   Época {epoca+1:2d}/{num_epocas} - "
          f"Perda: {perda_total/len(X_treino)*batch_size:.4f} - "
          f"Acurácia: {acuracia:.2%}")

# 5. TESTAR
print("\n🧪 Testando modelo...")
modelo.set_train(False)
X_teste_tensor = ms.Tensor(X_teste)
saidas = modelo(X_teste_tensor)
preds = np.argmax(saidas.asnumpy(), axis=1)

acuracia_teste = np.mean(preds == y_teste)
print(f"✅ Acurácia no teste: {acuracia_teste:.2%}")

# Mostrar exemplos
print("\n📋 Exemplos de predição:")
for i in range(min(5, len(X_teste))):
    pred_sinal = id_para_sinal[preds[i]]
    real_sinal = id_para_sinal[y_teste[i]]
    correto = "✓" if preds[i] == y_teste[i] else "✗"
    print(f"   {correto} Predito: {pred_sinal:10s} | Real: {real_sinal}")

# 6. SALVAR MODELO
print("\n💾 Salvando modelo...")
os.makedirs("models", exist_ok=True)

# Salvar checkpoint MindSpore
ms.save_checkpoint(modelo, "models/modelo_libras.ckpt")

# Salvar mapeamento
with open("models/mapeamento_sinais.pkl", "wb") as f:
    pickle.dump({
        "id_para_sinal": id_para_sinal,
        "sinal_para_id": sinal_para_id,
        "sinais": sinais_unicos
    }, f)

print("✅ Modelo salvo em: models/modelo_libras.ckpt")
print("✅ Mapeamento salvo em: models/mapeamento_sinais.pkl")

print("\n" + "="*60)
print("🎉 TREINAMENTO CONCLUÍDO!")
print("👉 Para testar em tempo real, execute:")
print("   python src\\reconhecer_tempo_real.py")
print("="*60)