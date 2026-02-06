# Power Procurement Strategy ML Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular ML pipeline for **power procurement strategy recommendation** in AI-focused data centers. The model takes into account real-world telemetry (GPU usage, workload type, site constraints) and outputs an optimal procurement plan (PPA %, Spot %, Battery %, timing, cost forecast).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Power Procurement ML Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │    Data      │    │   Power      │    │  Strategy    │                   │
│  │  Generator   │───▶│  Forecaster  │───▶│ Recommender  │                   │
│  │              │    │  (LSTM/TCN)  │    │  (XGBoost)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Synthetic   │    │   48-hour    │    │  PPA/Spot/   │                   │
│  │   Dataset    │    │ Power Trace  │    │   Battery    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
│                    ┌──────────────────┐                                     │
│                    │   RL Optimizer   │ (Optional)                          │
│                    │   (PPO/DQN)      │                                     │
│                    └──────────────────┘                                     │
│                                                                              │
│                    ┌──────────────────┐                                     │
│                    │    Streamlit     │                                     │
│                    │    Dashboard     │                                     │
│                    └──────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **📁 Synthetic Data Generation**: Creates realistic datasets with GPU utilization profiles, power traces, and market conditions
- **🧠 Power Demand Forecasting**: LSTM/TCN models predict 48-hour power consumption from GPU utilization
- **🎯 Procurement Strategy Recommendation**: XGBoost-based multi-output model recommends optimal energy mix
- **🧪 RL Policy Optimizer**: OpenAI Gym environment with PPO/DQN for dynamic procurement optimization
- **📊 Interactive Dashboard**: Streamlit web UI for real-time strategy analysis
- **📈 Comprehensive Evaluation**: Metrics, visualizations, and automated reporting

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd PDPPModel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Synthetic Dataset

```bash
python src/data_generator.py --samples 500 --output data/synthetic_dataset.csv
```

### Train Models

```bash
# Train power demand forecaster
python src/forecaster.py --data data/synthetic_dataset.csv --save models/forecaster

# Train procurement recommender
python src/recommender.py --data data/synthetic_dataset.csv --save models/recommender

# (Optional) Train RL policy optimizer
python src/rl_optimizer.py --save models/rl
```

### Run Dashboard

```bash
streamlit run app.py
```

### Jupyter Notebook Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

## 📂 Project Structure

```
PDPPModel/
├── config.yaml              # Configuration for all components
├── requirements.txt         # Python dependencies
├── app.py                   # Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic dataset generation
│   ├── forecaster.py        # LSTM/TCN power forecasting
│   ├── recommender.py       # XGBoost procurement recommender
│   ├── rl_optimizer.py      # RL-based policy optimizer
│   └── evaluate.py          # Metrics and reporting
├── notebooks/
│   └── demo.ipynb           # Interactive demo notebook
├── data/                    # Generated datasets
├── models/                  # Saved model checkpoints
└── outputs/                 # Visualizations and reports
```

## 📊 Data Schema

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `num_gpus` | int | Number of GPUs in cluster |
| `gpu_type` | categorical | A100, H100, MI300X |
| `rack_density_kw` | float | Power density per rack |
| `workload_type` | categorical | LLM_Training, Inference |
| `batch_size` | int | Training batch size |
| `seq_length` | int | Sequence length |
| `fp_precision` | string | fp16, bf16, fp32, int8 |
| `gpu_util_profile` | array[48] | Hourly GPU utilization |
| `power_trace_kw` | array[48] | Hourly power consumption |
| `burst_peak_kw` | float | Maximum power burst |
| `grid_capacity_mw` | float | Grid connection capacity |
| `renewable_target_pct` | float | Renewable energy target |
| `ppa_price_usd_mwh` | float | PPA contract price |
| `spot_price_avg_usd_mwh` | float | Spot market price |
| `battery_cost_usd_kwh` | float | Battery storage cost |

### Output Targets

| Target | Type | Description |
|--------|------|-------------|
| `recommended_mix` | JSON | {ppa, spot, battery} in MW |
| `contract_lead_time_months` | int | Recommended lead time |
| `forecasted_cost_usd_mwh` | float | Projected cost |

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
forecaster:
  model_type: "lstm"  # Options: lstm, tcn
  hidden_size: 128
  epochs: 100
  
recommender:
  model_type: "xgboost"  # Options: xgboost, random_forest
  n_estimators: 200
  
rl:
  algorithm: "PPO"  # Options: PPO, DQN
  total_timesteps: 50000
```

## 📈 Model Performance

### Power Forecaster
- **MAPE**: ~5-10%
- **RMSE**: ~15-25 kW (varies with cluster size)

### Procurement Recommender
- **Mix MAE**: ~0.5-1.0 MW
- **Cost MAPE**: ~8-15%
- **Lead Time MAE**: ~2-4 months

## 🎯 API Usage

### Python API

```python
from src.data_generator import PowerDataGenerator
from src.forecaster import PowerForecaster
from src.recommender import ProcurementRecommender

# Generate data
generator = PowerDataGenerator('config.yaml')
df = generator.generate_dataset(n_samples=100)

# Train and predict
forecaster = PowerForecaster('config.yaml')
forecaster.train(df)
power_pred = forecaster.predict(gpu_utilization)

recommender = ProcurementRecommender('config.yaml')
recommender.train(df)
strategy = recommender.predict_single(cluster_config)
```

### CLI Usage

```bash
# Generate data
python src/data_generator.py --samples 1000 --output data/large_dataset.csv

# Train with custom config
python src/forecaster.py --config custom_config.yaml --data data/dataset.csv

# Evaluate models
python src/evaluate.py --forecaster models/forecaster --recommender models/recommender
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for sustainable AI infrastructure
- Inspired by real-world data center power management challenges
- Uses PyTorch, XGBoost, and Stable-Baselines3
