# -*- coding: utf-8 -*-  
import mindspore as ms  
import mindspore.nn as nn  
import numpy as np  
  
class ModeloLibras(nn.Cell):  
    """Modelo para reconhecer sinais de Libras"""  
  
    def __init__(self, num_sinais=10):  
        super().__init__()  
        # 126 = 21 landmarks * 3 coordenadas * 2 maos  
        self.lstm = nn.LSTM(126, 128, num_layers=2, batch_first=True, bidirectional=True)  
        self.fc1 = nn.Dense(256, 64)  # 128*2 por ser bidirectional  
        self.relu = nn.ReLU()  
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Dense(64, num_sinais)  
  
    def construct(self, x):  
        # x shape: (batch, seq_len, 126)  
        lstm_out, (hn, cn) = self.lstm(x)  
        ultimo = lstm_out[:, -1, :]  # Ultimo time step  
        x = self.fc1(ultimo)  
        x = self.relu(x)  
        x = self.dropout(x)  
        return self.fc2(x)  
  
if __name__ == "__main__":  
    modelo = ModeloLibras()  
    dados_teste = ms.Tensor(np.random.randn(4, 30, 126).astype(np.float32))  
    saida = modelo(dados_teste)  
    print(f"Modelo criado com sucesso!")  
    print(f"Input shape: {dados_teste.shape}")  
    print(f"Output shape: {saida.shape}") 
