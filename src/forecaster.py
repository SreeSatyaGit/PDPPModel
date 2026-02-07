"""
Power Demand Forecasting Module
================================
Time-series forecasting using LSTM or TCN.
Improved scaling for wide power ranges.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional
import json
import yaml
from pathlib import Path
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
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
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
        self.model_type = fc['model_type']
        self.hidden = fc['hidden_size']
        self.layers = fc['num_layers']
        self.dropout = fc['dropout']
        self.lr = fc['learning_rate']
        self.epochs = fc['epochs']
        self.batch_size = fc['batch_size']
        self.patience = fc['early_stopping_patience']
        self.time_steps = self.config['data']['time_steps']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = RobustScaler()  # Better for wide ranges
        self.power_scaler = RobustScaler()
        self.use_log = True  # Use log transform for power
        self.power_offset = 0.1  # Offset for log transform
        self.history = {'train_loss': [], 'val_loss': []}
    
    def _create_model(self, meta_size=0):
        Model = LSTMForecaster if self.model_type == "lstm" else TCNForecaster
        return Model(1, self.hidden, self.layers, self.time_steps, self.dropout, meta_size).to(self.device)
    
    def _log_transform(self, x):
        """Apply log transform with offset to handle zeros."""
        return np.log1p(x + self.power_offset)
    
    def _inverse_log_transform(self, x):
        """Inverse log transform."""
        return np.expm1(x) - self.power_offset
    
    def prepare_data(self, df, include_meta=True, fit_scaler=True):
        # Parse GPU utilization
        gpu = np.array([json.loads(x) if isinstance(x, str) else x for x in df['gpu_util_profile']])
        
        # Parse power traces
        pwr = np.array([json.loads(x) if isinstance(x, str) else x for x in df['power_trace_kw']])
        
        # Apply log transform for better scaling of wide ranges
        if self.use_log:
            pwr_transformed = self._log_transform(pwr)
        else:
            pwr_transformed = pwr
        
        # Scale the data
        if fit_scaler:
            self.power_scaler.fit(pwr_transformed.reshape(-1, 1))
        pwr_scaled = self.power_scaler.transform(pwr_transformed.reshape(-1, 1)).reshape(pwr.shape)
        
        # Prepare metadata features
        meta = None
        if include_meta:
            # Use more relevant features
            cols = [c for c in ['num_gpus', 'num_nodes', 'num_cores', 'avg_power_kw', 'burst_peak_kw'] if c in df.columns]
            if not cols:
                cols = [c for c in ['num_gpus', 'rack_density_kw', 'burst_peak_kw'] if c in df.columns]
            if cols:
                meta_data = df[cols].fillna(0).values.astype(np.float32)
                if fit_scaler:
                    meta = self.scaler.fit_transform(meta_data)
                else:
                    meta = self.scaler.transform(meta_data)
        
        return gpu, pwr_scaled, meta
    
    def train(self, df, include_meta=True, verbose=True):
        # Prepare data
        gpu, pwr, meta = self.prepare_data(df, include_meta, fit_scaler=True)
        meta_size = meta.shape[1] if meta is not None else 0
        
        # Split data
        if meta is not None:
            X_tr, X_val, y_tr, y_val, m_tr, m_val = train_test_split(
                gpu, pwr, meta, test_size=0.1, random_state=42
            )
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(gpu, pwr, test_size=0.1, random_state=42)
            m_tr, m_val = None, None
        
        # Create data loaders
        train_dl = DataLoader(PowerDataset(X_tr, y_tr, m_tr), batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(PowerDataset(X_val, y_val, m_val), batch_size=self.batch_size)
        
        # Create model
        self.model = self._create_model(meta_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_loss, patience_cnt, best_state = float('inf'), 0, None
        
        for epoch in tqdm(range(self.epochs), disable=not verbose):
            # Training
            self.model.train()
            t_loss = sum(self._train_step(b, criterion, optimizer) for b in train_dl) / len(train_dl)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                v_loss = sum(self._val_step(b, criterion) for b in val_dl) / len(val_dl)
            
            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            
            scheduler.step(v_loss)
            
            if v_loss < best_loss:
                best_loss = v_loss
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
            
            if patience_cnt >= self.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        
        if meta is not None:
            if meta.ndim == 1:
                meta = meta.reshape(1, -1)
            m = torch.FloatTensor(self.scaler.transform(meta)).to(self.device)
        else:
            m = None
        
        with torch.no_grad():
            pred_scaled = self.model(x, m).cpu().numpy()
        
        # Inverse transform
        pred_log = self.power_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
        
        if self.use_log:
            pred = self._inverse_log_transform(pred_log)
        else:
            pred = pred_log
        
        # Ensure non-negative
        return np.maximum(pred, 0)
    
    def evaluate(self, df, include_meta=True):
        # Get original power values
        pwr_original = np.array([json.loads(x) if isinstance(x, str) else x for x in df['power_trace_kw']])
        
        # Get GPU profiles and metadata
        gpu = np.array([json.loads(x) if isinstance(x, str) else x for x in df['gpu_util_profile']])
        
        meta = None
        if include_meta:
            cols = [c for c in ['num_gpus', 'num_nodes', 'num_cores', 'avg_power_kw', 'burst_peak_kw'] if c in df.columns]
            if not cols:
                cols = [c for c in ['num_gpus', 'rack_density_kw', 'burst_peak_kw'] if c in df.columns]
            if cols:
                meta = df[cols].fillna(0).values.astype(np.float32)
        
        # Get predictions
        pred = self.predict(gpu, meta)
        actual = pwr_original
        
        # Calculate metrics (robust to small values)
        # Use symmetric MAPE to avoid division issues
        smape = np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8)) * 100
        
        # Standard MAPE with threshold
        threshold = 0.5  # Minimum 0.5 kW to avoid division by tiny values
        mask = np.mean(actual, axis=1) > threshold
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - pred[mask]) / (actual[mask] + 1e-8))) * 100
        else:
            mape = smape
        
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mae = np.mean(np.abs(actual - pred))
        
        # R-squared
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'MAPE': round(mape, 2),
            'sMAPE': round(smape, 2),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'R2': round(r2, 4)
        }
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'power_scaler_center': self.power_scaler.center_,
            'power_scaler_scale': self.power_scaler.scale_,
            'scaler_center': self.scaler.center_,
            'scaler_scale': self.scaler.scale_,
            'config': self.config,
            'use_log': self.use_log,
            'power_offset': self.power_offset
        }, f"{path}/forecaster.pt")
        print(f"Model saved to {path}/forecaster.pt")
    
    def load(self, path):
        ckpt = torch.load(f"{path}/forecaster.pt", map_location=self.device)
        self.config = ckpt['config']
        self.power_scaler.center_ = ckpt['power_scaler_center']
        self.power_scaler.scale_ = ckpt['power_scaler_scale']
        self.scaler.center_ = ckpt['scaler_center']
        self.scaler.scale_ = ckpt['scaler_scale']
        self.use_log = ckpt.get('use_log', True)
        self.power_offset = ckpt.get('power_offset', 0.1)
        self.model = self._create_model(len(self.scaler.center_))
        self.model.load_state_dict(ckpt['model'])
        print(f"Model loaded from {path}/forecaster.pt")

