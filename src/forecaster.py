"""
Power Demand Forecasting Module
================================
Time-series forecasting using LSTM or TCN.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


class PowerDataset(Dataset):
    def __init__(self, gpu_util: np.ndarray, power: np.ndarray, meta: Optional[np.ndarray] = None):
        self.gpu_util = torch.FloatTensor(gpu_util).unsqueeze(-1)
        self.power = torch.FloatTensor(power)
        self.meta = torch.FloatTensor(meta) if meta is not None else None
    
    def __len__(self): return len(self.gpu_util)
    
    def __getitem__(self, idx):
        if self.meta is not None:
            return self.gpu_util[idx], self.power[idx], self.meta[idx]
        return self.gpu_util[idx], self.power[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=48, dropout=0.2, meta_size=0):
        super().__init__()
        self.meta_size = meta_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + meta_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, meta=None):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        if meta is not None and self.meta_size > 0:
            h = torch.cat([h, meta], dim=-1)
        return self.fc(h)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pad = pad
    
    def forward(self, x):
        res = self.residual(x)
        out = self.dropout(self.relu(self.conv1(x)[:, :, :-self.pad] if self.pad else self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)[:, :, :-self.pad] if self.pad else self.conv2(out)))
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=4, output_size=48, dropout=0.2, meta_size=0):
        super().__init__()
        self.meta_size = meta_size
        layers = [TCNBlock(input_size if i == 0 else hidden_size, hidden_size, 3, 2**i, dropout) for i in range(num_layers)]
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(hidden_size + meta_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
    
    def forward(self, x, meta=None):
        out = self.tcn(x.transpose(1, 2)).mean(dim=-1)
        if meta is not None and self.meta_size > 0:
            out = torch.cat([out, meta], dim=-1)
        return self.fc(out)


class PowerForecaster:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        fc = self.config['forecaster']
        self.model_type, self.hidden, self.layers = fc['model_type'], fc['hidden_size'], fc['num_layers']
        self.dropout, self.lr, self.epochs = fc['dropout'], fc['learning_rate'], fc['epochs']
        self.batch_size, self.patience = fc['batch_size'], fc['early_stopping_patience']
        self.time_steps = self.config['data']['time_steps']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.scaler, self.power_scaler = None, StandardScaler(), StandardScaler()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def _create_model(self, meta_size=0):
        Model = LSTMForecaster if self.model_type == "lstm" else TCNForecaster
        return Model(1, self.hidden, self.layers, self.time_steps, self.dropout, meta_size).to(self.device)
    
    def prepare_data(self, df, include_meta=True):
        gpu = np.array([json.loads(x) if isinstance(x, str) else x for x in df['gpu_util_profile']])
        pwr = np.array([json.loads(x) if isinstance(x, str) else x for x in df['power_trace_kw']])
        self.power_scaler.fit(pwr.reshape(-1, 1))
        pwr_scaled = self.power_scaler.transform(pwr.reshape(-1, 1)).reshape(pwr.shape)
        meta = None
        if include_meta:
            cols = [c for c in ['num_gpus', 'rack_density_kw', 'burst_peak_kw'] if c in df.columns]
            if cols:
                meta = self.scaler.fit_transform(df[cols].values.astype(np.float32))
        return gpu, pwr_scaled, meta
    
    def train(self, df, include_meta=True, verbose=True):
        gpu, pwr, meta = self.prepare_data(df, include_meta)
        meta_size = meta.shape[1] if meta is not None else 0
        split_args = (gpu, pwr, meta) if meta is not None else (gpu, pwr)
        split = train_test_split(*split_args, test_size=0.1, random_state=42)
        if meta is not None:
            X_tr, X_val, y_tr, y_val, m_tr, m_val = split
        else:
            X_tr, X_val, y_tr, y_val = split
            m_tr, m_val = None, None
        
        train_dl = DataLoader(PowerDataset(X_tr, y_tr, m_tr), batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(PowerDataset(X_val, y_val, m_val), batch_size=self.batch_size)
        self.model = self._create_model(meta_size)
        criterion, optimizer = nn.MSELoss(), torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_loss, patience_cnt, best_state = float('inf'), 0, None
        
        for epoch in tqdm(range(self.epochs), disable=not verbose):
            self.model.train()
            t_loss = sum(self._train_step(b, criterion, optimizer) for b in train_dl) / len(train_dl)
            self.model.eval()
            with torch.no_grad():
                v_loss = sum(self._val_step(b, criterion) for b in val_dl) / len(val_dl)
            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            if v_loss < best_loss:
                best_loss, patience_cnt, best_state = v_loss, 0, {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
            if patience_cnt >= self.patience:
                break
        if best_state:
            self.model.load_state_dict(best_state)
        return self.history
    
    def _train_step(self, batch, criterion, optimizer):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        m = batch[2].to(self.device) if len(batch) == 3 else None
        optimizer.zero_grad()
        loss = criterion(self.model(x, m), y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def _val_step(self, batch, criterion):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        m = batch[2].to(self.device) if len(batch) == 3 else None
        return criterion(self.model(x, m), y).item()
    
    def predict(self, gpu_util, meta=None):
        self.model.eval()
        if gpu_util.ndim == 1:
            gpu_util = gpu_util.reshape(1, -1)
        x = torch.FloatTensor(gpu_util).unsqueeze(-1).to(self.device)
        m = torch.FloatTensor(self.scaler.transform(meta.reshape(1, -1) if meta.ndim == 1 else meta)).to(self.device) if meta is not None else None
        with torch.no_grad():
            pred = self.model(x, m).cpu().numpy()
        return self.power_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
    
    def evaluate(self, df, include_meta=True):
        gpu, pwr, meta = self.prepare_data(df, include_meta)
        pred = self.predict(gpu, meta)
        actual = self.power_scaler.inverse_transform(pwr.reshape(-1, 1)).reshape(pwr.shape)
        return {'MAPE': round(np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100, 2),
                'RMSE': round(np.sqrt(np.mean((actual - pred) ** 2)), 2),
                'MAE': round(np.mean(np.abs(actual - pred)), 2)}
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'power_scaler': (self.power_scaler.mean_, self.power_scaler.scale_),
                    'scaler': (self.scaler.mean_, self.scaler.scale_), 'config': self.config}, f"{path}/forecaster.pt")
    
    def load(self, path):
        ckpt = torch.load(f"{path}/forecaster.pt", map_location=self.device)
        self.config = ckpt['config']
        self.power_scaler.mean_, self.power_scaler.scale_ = ckpt['power_scaler']
        self.scaler.mean_, self.scaler.scale_ = ckpt['scaler']
        self.model = self._create_model(len(self.scaler.mean_))
        self.model.load_state_dict(ckpt['model'])
