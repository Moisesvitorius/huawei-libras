# -*- coding: utf-8 -*-
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import pickle
import os

print("="*60)
print("       HARMONYCARE - TREINAMENTO MINDSPORE")
print("="*60)

# Configurar contexto (IMPORTANTE!)
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")  # Usar PYNATIVE, não GRAPH

# 1. CARREGAR DADOS
print("\n[INFO] Carregando dados coletados...")
dados = []
rotulos_nomes = []

pasta_dados = "data"
for arquivo in os.listdir(pasta_dados):
    if arquivo.endswith(".pkl"):
        caminho = os.path.join(pasta_dados, arquivo)
        with open(caminho, "rb") as f:
            dados_sinal = pickle.load(f)
        
        sinal = dados_sinal["sinal"]
        frames = dados_sinal["frames"]
        
        # Garantir 30 frames
        seq_len = 30
        if len(frames) > seq_len:
            indices = np.linspace(0, len(frames)-1, seq_len, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < seq_len:
            ultimo_frame = frames[-1] if frames else [0]*126
            frames = frames + [ultimo_frame] * (seq_len - len(frames))
        
        dados.append(np.array(frames, dtype=np.float32))
        rotulos_nomes.append(sinal)

print(f"[OK] Dados carregados: {len(dados)} amostras")

# Converter rótulos
sinais_unicos = list(set(rotulos_nomes))
sinal_para_id = {sinal: i for i, sinal in enumerate(sinais_unicos)}
id_para_sinal = {i: sinal for i, sinal in enumerate(sinais_unicos)}

rotulos = [sinal_para_id[r] for r in rotulos_nomes]

print(f"[INFO] Sinais: {sinais_unicos}")

# 2. MODELO SIMPLIFICADO (evita problemas)
class ModeloLibrasSimples(nn.Cell):
    def __init__(self, num_classes=5):
        super().__init__()
        # Evitar LSTM que pode dar erro - usar CNN 1D
        self.conv1 = nn.Conv1d(126, 64, kernel_size=3, padding=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm1d(128)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(128 * 15, 64)  # 30 frames / 2 pooling = 15
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.fc2 = nn.Dense(64, num_classes)
    
    def construct(self, x):
        # x shape: (batch, seq_len=30, features=126)
        x = x.transpose(0, 2, 1)  # Para conv1d: (batch, features, seq_len)
        # print(f"Input T: {x.shape}")
        
        x = self.conv1(x)
        # print(f"Conv1: {x.shape}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        # print(f"Pool1: {x.shape}")
        
        x = self.conv2(x)
        # print(f"Conv2: {x.shape}")
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        # print(f"Flatten: {x.shape}")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 3. PREPARAR DATASET MindSpore
print("\n[INFO] Criando dataset MindSpore...")
X = np.array(dados)  # (n_amostras, 30, 126)
y = np.array(rotulos, dtype=np.int32)

# Embaralhar
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Converter para Tensor Dataset
def create_dataset(X_data, y_data, batch_size=4):
    def generator():
        for i in range(len(X_data)):
            yield X_data[i], y_data[i]
    
    dataset = ms.dataset.GeneratorDataset(
        generator, 
        column_names=["data", "label"],
        shuffle=True
    )
    return dataset.batch(batch_size)

# Dividir
split = int(0.8 * len(X))
train_dataset = create_dataset(X[:split], y[:split])
test_dataset = create_dataset(X[split:], y[split:])

print(f"   Treino: {split} amostras")
print(f"   Teste: {len(X)-split} amostras")

# 4. TREINAMENTO SIMPLIFICADO
print("\n[INFO] Iniciando treinamento...")
modelo = ModeloLibrasSimples(num_classes=len(sinais_unicos))

# Loss e otimizador
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(modelo.trainable_params(), learning_rate=0.001)

# Função forward
def forward_fn(data, label):
    logits = modelo(data)
    loss = loss_fn(logits, label)
    return loss, logits

# Grad function
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# Loop de treino
modelo.set_train()
num_epochs = 20

for epoch in range(num_epochs):
    losses = []
    correct = 0
    total = 0
    
    for batch, (data, label) in enumerate(train_dataset.create_tuple_iterator()):
        # Forward e backward
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        
        losses.append(loss.asnumpy())
        
        # Calcular acurácia
        preds = np.argmax(logits.asnumpy(), axis=1)
        correct += np.sum(preds == label.asnumpy())
        total += len(label)
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = np.mean(losses)
    
    print(f"   Época {epoch+1:2d}/{num_epochs} - "
          f"Loss: {avg_loss:.4f} - Acurácia: {accuracy:.2%}")

# 5. TESTAR
print("\n[INFO] Testando modelo...")
modelo.set_train(False)

test_correct = 0
test_total = 0

for data, label in test_dataset.create_tuple_iterator():
    logits = modelo(data)
    preds = np.argmax(logits.asnumpy(), axis=1)
    test_correct += np.sum(preds == label.asnumpy())
    test_total += len(label)

test_accuracy = test_correct / test_total if test_total > 0 else 0
print(f"[OK] Acuracia no teste: {test_accuracy:.2%}")

# 6. SALVAR
print("\n[INFO] Salvando modelo...")
os.makedirs("models", exist_ok=True)

# Salvar checkpoint
ms.save_checkpoint(modelo, "models/modelo_libras_mindspore.ckpt")

# Salvar mapeamento
import joblib
joblib.dump({
    "id_para_sinal": id_para_sinal,
    "sinal_para_id": sinal_para_id,
    "sinais": sinais_unicos
}, "models/mapeamento_sinais.pkl")

print("[OK] Modelo MindSpore salvo: models/modelo_libras_mindspore.ckpt")
print("[OK] Mapeamento salvo: models/mapeamento_sinais.pkl")

# 7. EXPORTAR PARA MINDIR (para inferência)
print("\n[INFO] Exportando modelo para MindIR...")
dummy_input = ms.Tensor(np.random.randn(1, 30, 126).astype(np.float32))
ms.export(modelo, dummy_input, file_name="models/modelo_libras", file_format="MINDIR")
print("[OK] Modelo exportado como MindIR")

print("\n" + "="*60)
print("TREINAMENTO MINDSPORE CONCLUIDO!")
print("Para testar em tempo real, execute:")
print("   python src\\reconhecer_mindspore.py")
print("="*60)