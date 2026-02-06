#!/usr/bin/env python3
"""
Power Procurement Strategy ML Pipeline - Main Entry Point
===========================================================

Usage:
    python main.py generate     # Generate synthetic dataset
    python main.py train        # Train all models
    python main.py evaluate     # Evaluate models
    python main.py predict      # Make predictions on sample
    python main.py dashboard    # Launch Streamlit dashboard
    python main.py demo         # Run full pipeline demo
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def generate_data(args):
    """Generate synthetic dataset."""
    from src.data_generator import PowerDataGenerator
    
    print("📁 Generating synthetic dataset...")
    generator = PowerDataGenerator(args.config)
    df = generator.generate_dataset(
        n_samples=args.samples,
        save_path=args.output
    )
    print(f"✅ Generated {len(df)} samples")
    print(f"   Saved to: {args.output}")
    return df


def train_models(args):
    """Train forecaster and recommender models."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.forecaster import PowerForecaster
    from src.recommender import ProcurementRecommender
    
    # Load data
    if not os.path.exists(args.data):
        print(f"⚠️  Data file not found: {args.data}")
        print("   Run 'python main.py generate' first")
        return
    
    print(f"📂 Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train forecaster
    print("\n🧠 Training Power Demand Forecaster...")
    forecaster = PowerForecaster(args.config)
    fc_history = forecaster.train(train_df)
    fc_metrics = forecaster.evaluate(test_df)
    forecaster.save("models/forecaster")
    print(f"   Metrics: {fc_metrics}")
    
    # Train recommender
    print("\n🎯 Training Procurement Recommender...")
    recommender = ProcurementRecommender(args.config)
    rec_metrics = recommender.train(train_df)
    recommender.save("models/recommender")
    
    print("\n✅ All models trained successfully!")
    return fc_metrics, rec_metrics


def evaluate_models(args):
    """Evaluate trained models."""
    import pandas as pd
    from src.forecaster import PowerForecaster
    from src.recommender import ProcurementRecommender
    from src.evaluate import PipelineEvaluator
    
    if not os.path.exists(args.data):
        print(f"⚠️  Data file not found: {args.data}")
        return
    
    print("📊 Evaluating models...")
    df = pd.read_csv(args.data)
    
    # Load and evaluate forecaster
    forecaster = PowerForecaster(args.config)
    forecaster.load("models/forecaster")
    fc_metrics = forecaster.evaluate(df)
    print(f"\n🧠 Forecaster Metrics: {fc_metrics}")
    
    # Load and evaluate recommender
    recommender = ProcurementRecommender(args.config)
    recommender.load("models/recommender")
    predictions = recommender.predict(df)
    print(f"\n🎯 Recommender Predictions Summary:")
    print(predictions.describe())
    
    # Generate report
    evaluator = PipelineEvaluator()
    evaluator.generate_report(
        fc_metrics,
        {'mix_mae': 0.5, 'cost_mape': 10},  # Placeholder
        predictions,
        save_path="reports/evaluation_report.md"
    )


def make_prediction(args):
    """Make prediction on sample cluster."""
    from src.recommender import ProcurementRecommender
    import numpy as np
    
    print("🔮 Making prediction for sample cluster...")
    
    # Sample cluster configuration
    sample = {
        'num_gpus': 1024,
        'gpu_type': 'H100',
        'rack_density_kw': 28.5,
        'workload_type': 'LLM_Training',
        'batch_size': 64,
        'seq_length': 4096,
        'fp_precision': 'bf16',
        'gpu_util_profile': [0.9 + np.random.normal(0, 0.03) for _ in range(48)],
        'power_trace_kw': [550 + np.random.normal(0, 25) for _ in range(48)],
        'burst_peak_kw': 720.0,
        'grid_capacity_mw': 80.0,
        'renewable_target_pct': 85.0,
        'onsite_gen_allowed': False,
        'ppa_price_usd_mwh': 52.0,
        'spot_price_avg_usd_mwh': 68.0,
        'battery_cost_usd_kwh': 175.0
    }
    
    recommender = ProcurementRecommender(args.config)
    recommender.load("models/recommender")
    
    result = recommender.predict_single(sample)
    
    print("\n" + "=" * 50)
    print("⚡ POWER PROCUREMENT RECOMMENDATION")
    print("=" * 50)
    print(f"\n📊 Cluster: {sample['num_gpus']} x {sample['gpu_type']}")
    print(f"   Workload: {sample['workload_type']}")
    print(f"\n🎯 Recommended Mix:")
    for k, v in result['recommended_mix'].items():
        pct = result['mix_percentages'][k]
        print(f"   {k.upper():8s}: {v:6.3f} MW ({pct:.1f}%)")
    print(f"\n💰 Forecasted Cost: ${result['forecasted_cost_usd_mwh']:.2f}/MWh")
    print(f"📅 Lead Time: {result['contract_lead_time_months']} months")
    print("=" * 50)


def run_dashboard(args):
    """Launch Streamlit dashboard."""
    import subprocess
    print("🚀 Launching Streamlit dashboard...")
    subprocess.run(["streamlit", "run", "app.py"])


def run_demo(args):
    """Run full pipeline demo."""
    print("🎬 Running full pipeline demo...\n")
    
    # Generate data
    class Args:
        config = args.config
        samples = 500
        output = "data/demo_dataset.csv"
        data = "data/demo_dataset.csv"
    
    demo_args = Args()
    
    generate_data(demo_args)
    print()
    train_models(demo_args)
    print()
    make_prediction(demo_args)
    
    print("\n✅ Demo complete!")
    print("   Run 'python main.py dashboard' to explore interactively")


def main():
    parser = argparse.ArgumentParser(
        description="Power Procurement Strategy ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument("--samples", type=int, default=500, help="Number of samples")
    gen_parser.add_argument("--output", type=str, default="data/synthetic_dataset.csv")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--data", type=str, default="data/synthetic_dataset.csv")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("--data", type=str, default="data/synthetic_dataset.csv")
    
    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Make prediction")
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch dashboard")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run full demo")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_data(args)
    elif args.command == "train":
        train_models(args)
    elif args.command == "evaluate":
        evaluate_models(args)
    elif args.command == "predict":
        make_prediction(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
