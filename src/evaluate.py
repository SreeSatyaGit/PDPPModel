"""
Evaluation and Metrics Module
==============================
Comprehensive evaluation utilities for all pipeline components.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import json


class PipelineEvaluator:
    """Unified evaluation for forecasting and recommendation models."""
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAPE': round(mape, 2),
            'R2': round(r2, 4)
        }
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residuals", save_path: Optional[str] = None):
        """Plot residual analysis."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Residual distribution
        axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--')
        axes[0].set_xlabel('Residual')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{title} - Distribution')
        
        # Residual vs Predicted
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residual')
        axes[1].set_title(f'{title} - Residuals vs Predicted')
        
        # Actual vs Predicted
        axes[2].scatter(y_true, y_pred, alpha=0.5)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        axes[2].set_xlabel('Actual')
        axes[2].set_ylabel('Predicted')
        axes[2].set_title(f'{title} - Actual vs Predicted')
        axes[2].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_time_series_comparison(
        actual: np.ndarray, 
        predicted: np.ndarray,
        sample_idx: int = 0,
        title: str = "Power Trace Comparison",
        save_path: Optional[str] = None
    ):
        """Plot actual vs predicted time series."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        t = np.arange(len(actual[sample_idx]))
        
        # Main comparison
        axes[0].plot(t, actual[sample_idx], 'b-', label='Actual', linewidth=2)
        axes[0].plot(t, predicted[sample_idx], 'r--', label='Predicted', linewidth=2)
        axes[0].fill_between(t, actual[sample_idx], predicted[sample_idx], alpha=0.3)
        axes[0].set_xlabel('Hour')
        axes[0].set_ylabel('Power (kW)')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error plot
        error = actual[sample_idx] - predicted[sample_idx]
        axes[1].bar(t, error, color='gray', alpha=0.7)
        axes[1].axhline(0, color='black', linestyle='-')
        axes[1].set_xlabel('Hour')
        axes[1].set_ylabel('Error (kW)')
        axes[1].set_title('Prediction Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_procurement_mix(
        predictions: pd.DataFrame,
        n_samples: int = 10,
        save_path: Optional[str] = None
    ):
        """Visualize recommended procurement mix distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Stacked bar chart for samples
        sample_data = predictions.head(n_samples)
        x = np.arange(len(sample_data))
        width = 0.6
        
        axes[0].bar(x, sample_data['ppa_mw'], width, label='PPA', color='#2ecc71')
        axes[0].bar(x, sample_data['spot_mw'], width, bottom=sample_data['ppa_mw'], 
                   label='Spot', color='#e74c3c')
        axes[0].bar(x, sample_data['battery_mw'], width, 
                   bottom=sample_data['ppa_mw'] + sample_data['spot_mw'],
                   label='Battery', color='#3498db')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Power (MW)')
        axes[0].set_title('Recommended Procurement Mix')
        axes[0].legend()
        axes[0].set_xticks(x)
        
        # Pie chart for average mix
        avg_mix = [predictions['ppa_mw'].mean(), 
                   predictions['spot_mw'].mean(), 
                   predictions['battery_mw'].mean()]
        labels = ['PPA', 'Spot', 'Battery']
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        axes[1].pie(avg_mix, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Average Mix Distribution')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_cost_analysis(
        predictions: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Analyze cost predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cost distribution
        axes[0].hist(predictions['forecasted_cost_usd_mwh'], bins=30, 
                    edgecolor='black', alpha=0.7, color='#9b59b6')
        axes[0].axvline(predictions['forecasted_cost_usd_mwh'].mean(), 
                       color='red', linestyle='--', label=f"Mean: ${predictions['forecasted_cost_usd_mwh'].mean():.2f}")
        axes[0].set_xlabel('Cost ($/MWh)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Forecasted Cost Distribution')
        axes[0].legend()
        
        # Lead time vs Cost
        axes[1].scatter(predictions['contract_lead_time_months'], 
                       predictions['forecasted_cost_usd_mwh'], alpha=0.5)
        axes[1].set_xlabel('Contract Lead Time (months)')
        axes[1].set_ylabel('Forecasted Cost ($/MWh)')
        axes[1].set_title('Lead Time vs Cost')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def generate_report(
        forecaster_metrics: Dict,
        recommender_metrics: Dict,
        predictions: pd.DataFrame,
        save_path: str = "reports/evaluation_report.md"
    ):
        """Generate markdown evaluation report."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# Power Procurement ML Pipeline - Evaluation Report

## 📊 Model Performance Summary

### Power Demand Forecaster
| Metric | Value |
|--------|-------|
| MAPE | {forecaster_metrics.get('MAPE', 'N/A')}% |
| RMSE | {forecaster_metrics.get('RMSE', 'N/A')} kW |
| MAE | {forecaster_metrics.get('MAE', 'N/A')} kW |

### Procurement Recommender
| Metric | Value |
|--------|-------|
| Mix MAE | {recommender_metrics.get('mix_mae', 'N/A'):.4f} MW |
| Mix RMSE | {recommender_metrics.get('mix_rmse', 'N/A'):.4f} MW |
| Lead Time MAE | {recommender_metrics.get('lead_mae', 'N/A'):.2f} months |
| Cost MAE | ${recommender_metrics.get('cost_mae', 'N/A'):.2f}/MWh |
| Cost MAPE | {recommender_metrics.get('cost_mape', 'N/A'):.2f}% |

## 📈 Prediction Statistics

### Recommended Mix Distribution
| Component | Mean (MW) | Std (MW) | Min (MW) | Max (MW) |
|-----------|-----------|----------|----------|----------|
| PPA | {predictions['ppa_mw'].mean():.3f} | {predictions['ppa_mw'].std():.3f} | {predictions['ppa_mw'].min():.3f} | {predictions['ppa_mw'].max():.3f} |
| Spot | {predictions['spot_mw'].mean():.3f} | {predictions['spot_mw'].std():.3f} | {predictions['spot_mw'].min():.3f} | {predictions['spot_mw'].max():.3f} |
| Battery | {predictions['battery_mw'].mean():.3f} | {predictions['battery_mw'].std():.3f} | {predictions['battery_mw'].min():.3f} | {predictions['battery_mw'].max():.3f} |

### Cost Forecast
- **Mean Cost**: ${predictions['forecasted_cost_usd_mwh'].mean():.2f}/MWh
- **Std Cost**: ${predictions['forecasted_cost_usd_mwh'].std():.2f}/MWh
- **Range**: ${predictions['forecasted_cost_usd_mwh'].min():.2f} - ${predictions['forecasted_cost_usd_mwh'].max():.2f}/MWh

### Lead Time
- **Mean**: {predictions['contract_lead_time_months'].mean():.1f} months
- **Range**: {predictions['contract_lead_time_months'].min()} - {predictions['contract_lead_time_months'].max()} months

---
*Report generated automatically by the Power Procurement ML Pipeline*
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {save_path}")
        return report


def main():
    """Demo evaluation functionality."""
    # Generate sample data for demonstration
    np.random.seed(42)
    n = 100
    
    y_true = np.random.uniform(50, 150, n)
    y_pred = y_true + np.random.normal(0, 10, n)
    
    evaluator = PipelineEvaluator()
    
    print("📊 Regression Metrics:")
    metrics = evaluator.regression_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Create sample predictions DataFrame
    predictions = pd.DataFrame({
        'ppa_mw': np.random.uniform(5, 20, n),
        'spot_mw': np.random.uniform(2, 10, n),
        'battery_mw': np.random.uniform(1, 5, n),
        'forecasted_cost_usd_mwh': np.random.uniform(40, 80, n),
        'contract_lead_time_months': np.random.randint(3, 24, n)
    })
    
    evaluator.plot_procurement_mix(predictions)
    evaluator.plot_cost_analysis(predictions)


if __name__ == "__main__":
    main()
