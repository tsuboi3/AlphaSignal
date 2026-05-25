"""
Transformerモデルモジュール (PyTorch版)
※ AVX非対応環境でも動作するように、TensorFlowからPyTorchに移行しました。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerModel(nn.Module):
    def __init__(self, input_shape, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=2, mlp_units=[128, 64], dropout=0.1):
        super(TransformerModel, self).__init__()
        seq_len, num_features = input_shape
        
        # 入力射影
        self.input_projection = nn.Linear(num_features, head_size * num_heads)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_size * num_heads,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_blocks)
        
        # MLP Layers
        layers = []
        curr_dim = head_size * num_heads
        for units in mlp_units:
            layers.append(nn.Linear(curr_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = units
        
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        # Global Average Pooling
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x

    def predict(self, X, verbose=0):
        """Keras互換のpredictメソッド"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        X_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            preds = self(X_t).cpu().numpy()
        return preds

def build_transformer_model(
    input_shape: tuple,
    head_size: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_transformer_blocks: int = 2,
    mlp_units: list = None,
    dropout: float = 0.1,
) -> TransformerModel:
    """PyTorch版Transformerモデルの構築"""
    if mlp_units is None:
        mlp_units = [128, 64]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout
    ).to(device)
    return model

def train_transformer(
    model: TransformerModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> TransformerModel:
    """PyTorch版Transformerモデルの学習"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    # データを分割
    n_samples = len(X_train)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val
    
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    
    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
    val_ds = TensorDataset(X_t[n_train:], y_t[n_train:])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False) # 時系列なのでshuffle=False
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= n_train
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= n_val
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # モデルの重みを保存（メモリ上に保持）
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            break
            
    model.load_state_dict(best_model_state)
    print(f"  → 学習完了 | val_loss: {best_val_loss:.6f}")
    return model

# predictメソッドの追加（Modelインターフェースに合わせる）
def predict_wrapper(model, X, verbose=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds

# Modelクラスにpredictメソッドを生やすための工夫
# 本来はラッパークラスを作るべきだが、既存コードとの互換性のために動的に追加するか
# main.py側で吸収する
