"""
Procurement Strategy Recommender
================================
Multi-output XGBoost/sklearn model for power procurement recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import json
import yaml
import joblib
from pathlib import Path


class ProcurementRecommender:
    """XGBoost-based procurement strategy recommender."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        rec = self.config['recommender']
        self.model_type = rec['model_type']
        self.n_estimators = rec['n_estimators']
        self.max_depth = rec['max_depth']
        self.lr = rec['learning_rate']
        self.early_stopping = rec['early_stopping_rounds']
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.mix_model = None
        self.lead_model = None
        self.cost_model = None
        self.feature_names = []
    
    def _create_base_model(self):
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.lr, random_state=42, n_jobs=-1
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42, n_jobs=-1
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.lr, random_state=42
            )
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and encode features from DataFrame."""
        features = []
        
        # Numeric features
        num_cols = ['num_gpus', 'rack_density_kw', 'burst_peak_kw', 'grid_capacity_mw',
                    'renewable_target_pct', 'ppa_price_usd_mwh', 'spot_price_avg_usd_mwh',
                    'battery_cost_usd_kwh', 'batch_size', 'seq_length']
        for col in num_cols:
            if col in df.columns:
                features.append(df[col].values.reshape(-1, 1))
                if col not in self.feature_names:
                    self.feature_names.append(col)
        
        # Categorical features
        cat_cols = ['gpu_type', 'workload_type', 'fp_precision']
        for col in cat_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    encoded = self.label_encoders[col].fit_transform(df[col])
                else:
                    encoded = self.label_encoders[col].transform(df[col])
                features.append(encoded.reshape(-1, 1))
                if col not in self.feature_names:
                    self.feature_names.append(col)
        
        # Boolean features
        if 'onsite_gen_allowed' in df.columns:
            features.append(df['onsite_gen_allowed'].astype(int).values.reshape(-1, 1))
            if 'onsite_gen_allowed' not in self.feature_names:
                self.feature_names.append('onsite_gen_allowed')
        
        # Time-series statistics
        if 'gpu_util_profile' in df.columns:
            utils = [json.loads(x) if isinstance(x, str) else x for x in df['gpu_util_profile']]
            util_stats = np.array([[np.mean(u), np.std(u), np.max(u), np.min(u)] for u in utils])
            features.append(util_stats)
            for name in ['util_mean', 'util_std', 'util_max', 'util_min']:
                if name not in self.feature_names:
                    self.feature_names.append(name)
        
        if 'power_trace_kw' in df.columns:
            powers = [json.loads(x) if isinstance(x, str) else x for x in df['power_trace_kw']]
            power_stats = np.array([[np.mean(p), np.std(p), np.max(p)] for p in powers])
            features.append(power_stats)
            for name in ['power_mean', 'power_std', 'power_max']:
                if name not in self.feature_names:
                    self.feature_names.append(name)
        
        return np.hstack(features)
    
    def _extract_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract target values from DataFrame."""
        # Recommended mix (PPA, Spot, Battery)
        mixes = [json.loads(x) if isinstance(x, str) else x for x in df['recommended_mix']]
        mix_array = np.array([[m['ppa'], m['spot'], m['battery']] for m in mixes])
        
        # Lead time
        lead_times = df['contract_lead_time_months'].values.reshape(-1, 1)
        
        # Cost
        costs = df['forecasted_cost_usd_mwh'].values.reshape(-1, 1)
        
        return mix_array, lead_times, costs
    
    def train(self, df: pd.DataFrame, verbose: bool = True) -> Dict[str, float]:
        """Train the recommender models."""
        # Extract features and targets
        X = self._extract_features(df)
        y_mix, y_lead, y_cost = self._extract_targets(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_tr, X_val, mix_tr, mix_val, lead_tr, lead_val, cost_tr, cost_val = train_test_split(
            X_scaled, y_mix, y_lead, y_cost, test_size=0.2, random_state=42
        )
        
        # Train multi-output model for procurement mix
        if verbose:
            print("Training procurement mix model...")
        self.mix_model = MultiOutputRegressor(self._create_base_model())
        self.mix_model.fit(X_tr, mix_tr)
        
        # Train lead time model
        if verbose:
            print("Training lead time model...")
        self.lead_model = self._create_base_model()
        self.lead_model.fit(X_tr, lead_tr.ravel())
        
        # Train cost model
        if verbose:
            print("Training cost model...")
        self.cost_model = self._create_base_model()
        self.cost_model.fit(X_tr, cost_tr.ravel())
        
        # Evaluate
        return self._evaluate(X_val, mix_val, lead_val, cost_val, verbose)
    
    def _evaluate(self, X, mix_true, lead_true, cost_true, verbose=True) -> Dict[str, float]:
        """Evaluate model performance."""
        mix_pred = self.mix_model.predict(X)
        lead_pred = self.lead_model.predict(X)
        cost_pred = self.cost_model.predict(X)
        
        metrics = {
            'mix_mae': np.mean(np.abs(mix_true - mix_pred)),
            'mix_rmse': np.sqrt(np.mean((mix_true - mix_pred) ** 2)),
            'lead_mae': np.mean(np.abs(lead_true.ravel() - lead_pred)),
            'cost_mae': np.mean(np.abs(cost_true.ravel() - cost_pred)),
            'cost_mape': np.mean(np.abs((cost_true.ravel() - cost_pred) / (cost_true.ravel() + 1e-8))) * 100
        }
        
        if verbose:
            print(f"\n📊 Evaluation Metrics:")
            print(f"  Mix MAE: {metrics['mix_mae']:.4f} MW")
            print(f"  Mix RMSE: {metrics['mix_rmse']:.4f} MW")
            print(f"  Lead Time MAE: {metrics['lead_mae']:.2f} months")
            print(f"  Cost MAE: ${metrics['cost_mae']:.2f}/MWh")
            print(f"  Cost MAPE: {metrics['cost_mape']:.2f}%")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict procurement strategy for new data."""
        X = self._extract_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        mix_pred = self.mix_model.predict(X_scaled)
        lead_pred = self.lead_model.predict(X_scaled)
        cost_pred = self.cost_model.predict(X_scaled)
        
        # Build results DataFrame
        results = pd.DataFrame({
            'ppa_mw': np.maximum(0, mix_pred[:, 0]),
            'spot_mw': np.maximum(0, mix_pred[:, 1]),
            'battery_mw': np.maximum(0, mix_pred[:, 2]),
            'contract_lead_time_months': np.round(lead_pred).astype(int),
            'forecasted_cost_usd_mwh': cost_pred
        })
        
        # Add percentages
        total = results['ppa_mw'] + results['spot_mw'] + results['battery_mw']
        results['ppa_pct'] = (results['ppa_mw'] / total * 100).round(1)
        results['spot_pct'] = (results['spot_mw'] / total * 100).round(1)
        results['battery_pct'] = (results['battery_mw'] / total * 100).round(1)
        
        return results
    
    def predict_single(self, sample: Dict) -> Dict:
        """Predict for a single sample (dict input)."""
        df = pd.DataFrame([sample])
        result = self.predict(df).iloc[0]
        return {
            'recommended_mix': {
                'ppa': round(result['ppa_mw'], 3),
                'spot': round(result['spot_mw'], 3),
                'battery': round(result['battery_mw'], 3)
            },
            'mix_percentages': {
                'ppa': result['ppa_pct'],
                'spot': result['spot_pct'],
                'battery': result['battery_pct']
            },
            'contract_lead_time_months': int(result['contract_lead_time_months']),
            'forecasted_cost_usd_mwh': round(result['forecasted_cost_usd_mwh'], 2)
        }
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from models."""
        if self.model_type != "xgboost" and self.model_type != "random_forest":
            return pd.DataFrame()
        
        # Average importance across mix model estimators
        if hasattr(self.mix_model.estimators_[0], 'feature_importances_'):
            imp = np.mean([e.feature_importances_ for e in self.mix_model.estimators_], axis=0)
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': imp
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save models and preprocessing artifacts."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'mix_model': self.mix_model,
            'lead_model': self.lead_model,
            'cost_model': self.cost_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'config': self.config
        }, f"{path}/recommender.joblib")
        
        print(f"Models saved to {path}/recommender.joblib")
    
    def load(self, path: str):
        """Load models and preprocessing artifacts."""
        data = joblib.load(f"{path}/recommender.joblib")
        
        self.mix_model = data['mix_model']
        self.lead_model = data['lead_model']
        self.cost_model = data['cost_model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.config = data['config']
        
        print(f"Models loaded from {path}/recommender.joblib")


def main():
    """Train and evaluate the recommender."""
    import argparse
    from data_generator import PowerDataGenerator
    
    parser = argparse.ArgumentParser(description="Train procurement recommender")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--save", type=str, default="models/recommender")
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.data:
        df = pd.read_csv(args.data)
    else:
        print("Generating synthetic dataset...")
        generator = PowerDataGenerator(args.config)
        df = generator.generate_dataset()
    
    # Train
    print("\n🔧 Training recommender...")
    recommender = ProcurementRecommender(args.config)
    metrics = recommender.train(df)
    
    # Feature importance
    print("\n📈 Top Features:")
    print(recommender.feature_importance().head(10))
    
    # Sample prediction
    print("\n🎯 Sample Prediction:")
    sample = df.iloc[0].to_dict()
    pred = recommender.predict_single(sample)
    print(f"  Mix: {pred['recommended_mix']}")
    print(f"  Lead Time: {pred['contract_lead_time_months']} months")
    print(f"  Cost: ${pred['forecasted_cost_usd_mwh']}/MWh")
    
    # Save
    recommender.save(args.save)


if __name__ == "__main__":
    main()
