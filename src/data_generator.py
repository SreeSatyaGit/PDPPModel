"""
Synthetic Data Generator for Power Procurement Strategy Model
==============================================================

Generates realistic synthetic datasets for training the procurement
strategy recommendation model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import yaml


class PowerDataGenerator:
    """
    Generates synthetic power procurement datasets with realistic
    characteristics for AI data center workloads.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the data generator with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_samples = self.config['data']['num_samples']
        self.time_steps = self.config['data']['time_steps']
        self.random_seed = self.config['data']['random_seed']
        
        np.random.seed(self.random_seed)
        
        self.gpu_types = self.config['gpu_types']
        self.workload_types = self.config['workload_types']
        self.gpu_power_specs = self.config['gpu_power_specs']
    
    def _generate_gpu_utilization_profile(
        self, 
        workload_type: str, 
        num_gpus: int
    ) -> np.ndarray:
        """
        Generate realistic GPU utilization profiles (48 hourly values).
        
        Training workloads: sustained high utilization with periodic dips
        Inference workloads: variable utilization following demand patterns
        """
        t = np.arange(self.time_steps)
        
        if workload_type == "LLM_Training":
            # Sustained high utilization (80-98%) with checkpointing dips
            base_util = np.random.uniform(0.85, 0.95)
            noise = np.random.normal(0, 0.03, self.time_steps)
            profile = np.clip(base_util + noise, 0.75, 0.99)
            
            # Add periodic checkpointing dips
            checkpoint_intervals = np.random.choice([4, 6, 8, 12])
            for i in range(0, self.time_steps, checkpoint_intervals):
                if i < self.time_steps:
                    profile[i] = np.random.uniform(0.3, 0.5)
        else:  # Inference
            # Variable utilization following diurnal pattern
            base_pattern = 0.5 + 0.3 * np.sin(2 * np.pi * (t - 6) / 24)
            noise = np.random.normal(0, 0.1, self.time_steps)
            spikes = np.random.choice([0, 0.2, 0.3], self.time_steps, p=[0.7, 0.2, 0.1])
            profile = np.clip(base_pattern + noise + spikes, 0.1, 0.98)
        
        return profile
    
    def _generate_power_trace(
        self, 
        gpu_util_profile: np.ndarray,
        num_gpus: int,
        gpu_type: str,
        rack_density_kw: float
    ) -> Tuple[np.ndarray, float]:
        """
        Generate power trace based on GPU utilization and specifications.
        
        Returns:
            power_trace_kw: Array of hourly power values
            burst_peak_kw: Maximum power burst observed
        """
        specs = self.gpu_power_specs[gpu_type]
        
        # Base power scales with utilization
        base_power = specs['min_power'] + (specs['tdp'] - specs['min_power']) * gpu_util_profile
        
        # Scale by number of GPUs
        total_gpu_power = base_power * num_gpus
        
        # Add overhead (cooling, networking, storage) - typically 20-40% PUE factor
        pue_factor = np.random.uniform(1.2, 1.5)
        
        # Add noise and transient spikes
        noise = np.random.normal(0, 0.02 * total_gpu_power.mean(), self.time_steps)
        spikes = np.random.choice(
            [0, 0.05, 0.1], 
            self.time_steps, 
            p=[0.8, 0.15, 0.05]
        ) * total_gpu_power.mean()
        
        power_trace = (total_gpu_power * pue_factor) + noise + spikes
        power_trace = np.maximum(power_trace, 0.1)  # Minimum power floor
        
        # Calculate burst peak with margin
        burst_peak_kw = float(np.max(power_trace) * np.random.uniform(1.05, 1.15))
        
        return power_trace, burst_peak_kw
    
    def _generate_recommended_mix(
        self,
        burst_peak_kw: float,
        avg_power_kw: float,
        renewable_target_pct: float,
        ppa_price: float,
        spot_price: float,
        battery_cost: float,
        grid_capacity_mw: float,
        onsite_gen_allowed: bool
    ) -> Dict[str, float]:
        """
        Generate realistic procurement mix recommendation based on inputs.
        
        This implements a simplified heuristic that a real optimization
        would produce.
        """
        # Convert to MW for consistency
        avg_power_mw = avg_power_kw / 1000
        peak_power_mw = burst_peak_kw / 1000
        
        # Calculate base PPA (for stable baseload, meeting renewable targets)
        min_ppa_for_renewables = avg_power_mw * (renewable_target_pct / 100)
        
        # Economic comparison: PPA vs Spot
        ppa_attractive = ppa_price < spot_price * 1.1  # PPA attractive if cheaper
        
        if ppa_attractive:
            # Favor PPA for baseload
            ppa_mw = max(min_ppa_for_renewables, avg_power_mw * np.random.uniform(0.6, 0.8))
        else:
            # Minimize PPA to renewable requirements
            ppa_mw = min_ppa_for_renewables * np.random.uniform(1.0, 1.2)
        
        # Spot for variable load
        spot_mw = max(0, avg_power_mw - ppa_mw) + np.random.uniform(0, 0.1) * avg_power_mw
        
        # Battery for peak shaving and arbitrage
        peak_delta = peak_power_mw - avg_power_mw
        if battery_cost < 200:  # Economic threshold for battery
            battery_mw = min(peak_delta * np.random.uniform(0.3, 0.6), 0.2 * avg_power_mw)
        else:
            battery_mw = peak_delta * np.random.uniform(0.1, 0.3)
        
        # Ensure within grid capacity
        total = ppa_mw + spot_mw + battery_mw
        if total > grid_capacity_mw:
            scale = grid_capacity_mw / total * 0.95
            ppa_mw *= scale
            spot_mw *= scale
            battery_mw *= scale
        
        return {
            "ppa": round(ppa_mw, 3),
            "spot": round(spot_mw, 3),
            "battery": round(battery_mw, 3)
        }
    
    def _calculate_forecasted_cost(
        self,
        recommended_mix: Dict[str, float],
        ppa_price: float,
        spot_price: float,
        battery_cost: float
    ) -> float:
        """
        Calculate forecasted cost per MWh for the recommended mix.
        """
        total_mw = recommended_mix['ppa'] + recommended_mix['spot']
        if total_mw == 0:
            return spot_price
        
        # Weighted average cost
        ppa_cost = recommended_mix['ppa'] * ppa_price
        spot_cost = recommended_mix['spot'] * spot_price
        # Battery adds amortized cost
        battery_amortized = recommended_mix['battery'] * battery_cost * 0.1  # 10% annual amortization
        
        total_cost = (ppa_cost + spot_cost + battery_amortized) / total_mw
        
        # Add noise
        return round(total_cost * np.random.uniform(0.95, 1.05), 2)
    
    def _generate_contract_lead_time(
        self,
        recommended_mix: Dict[str, float],
        grid_capacity_mw: float
    ) -> int:
        """
        Estimate contract lead time based on procurement complexity.
        """
        ppa_ratio = recommended_mix['ppa'] / (recommended_mix['ppa'] + recommended_mix['spot'] + 0.001)
        
        if ppa_ratio > 0.7:
            # Heavy PPA requires longer negotiation
            base_time = np.random.randint(12, 24)
        elif ppa_ratio > 0.4:
            base_time = np.random.randint(6, 18)
        else:
            # Spot-heavy is faster
            base_time = np.random.randint(3, 12)
        
        # Large capacity adds time
        if grid_capacity_mw > 50:
            base_time += np.random.randint(3, 12)
        
        return int(min(base_time, 36))  # Cap at 3 years
    
    def generate_single_sample(self) -> Dict:
        """
        Generate a single training sample.
        """
        # Cluster specifications
        num_gpus = int(np.random.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192]))
        gpu_type = np.random.choice(self.gpu_types)
        workload_type = np.random.choice(self.workload_types)
        
        # Rack density correlates with GPU type
        base_density = {'A100': 15, 'H100': 25, 'MI300X': 30}
        rack_density_kw = base_density[gpu_type] * np.random.uniform(0.8, 1.2)
        
        # Training parameters (relevant for LLM training)
        if workload_type == "LLM_Training":
            batch_size = int(np.random.choice([8, 16, 32, 64, 128, 256]))
            seq_length = int(np.random.choice([512, 1024, 2048, 4096, 8192]))
            fp_precision = np.random.choice(["fp16", "bf16", "fp32"])
        else:
            batch_size = int(np.random.choice([1, 2, 4, 8, 16, 32]))
            seq_length = int(np.random.choice([128, 256, 512, 1024, 2048]))
            fp_precision = np.random.choice(["fp16", "int8", "int4"])
        
        # Generate utilization and power profiles
        gpu_util_profile = self._generate_gpu_utilization_profile(workload_type, num_gpus)
        power_trace_kw, burst_peak_kw = self._generate_power_trace(
            gpu_util_profile, num_gpus, gpu_type, rack_density_kw
        )
        
        # Grid and site constraints
        grid_capacity_mw = max(burst_peak_kw / 1000 * 1.5, np.random.uniform(10, 200))
        renewable_target_pct = float(np.random.uniform(50, 100))
        onsite_gen_allowed = bool(np.random.choice([True, False], p=[0.3, 0.7]))
        
        # Market prices (realistic ranges)
        ppa_price_usd_mwh = float(np.random.uniform(30, 80))
        spot_price_avg_usd_mwh = float(np.random.uniform(25, 120))
        battery_cost_usd_kwh = float(np.random.uniform(100, 300))
        
        # Generate labels
        avg_power_kw = float(np.mean(power_trace_kw))
        recommended_mix = self._generate_recommended_mix(
            burst_peak_kw, avg_power_kw, renewable_target_pct,
            ppa_price_usd_mwh, spot_price_avg_usd_mwh, battery_cost_usd_kwh,
            grid_capacity_mw, onsite_gen_allowed
        )
        
        forecasted_cost = self._calculate_forecasted_cost(
            recommended_mix, ppa_price_usd_mwh, spot_price_avg_usd_mwh, battery_cost_usd_kwh
        )
        
        contract_lead_time = self._generate_contract_lead_time(recommended_mix, grid_capacity_mw)
        
        # Vendor recommendation based on characteristics
        vendors = ["Enel Green Power", "NextEra Energy", "Ørsted", "EDF Renewables", 
                   "Iberdrola", "Shell Energy", "Engie", "AES Corporation"]
        vendor = np.random.choice(vendors)
        
        return {
            # Input features
            "num_gpus": num_gpus,
            "gpu_type": gpu_type,
            "rack_density_kw": round(rack_density_kw, 2),
            "workload_type": workload_type,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "fp_precision": fp_precision,
            "gpu_util_profile": gpu_util_profile.tolist(),
            "power_trace_kw": power_trace_kw.tolist(),
            "burst_peak_kw": round(burst_peak_kw, 2),
            "grid_capacity_mw": round(grid_capacity_mw, 2),
            "renewable_target_pct": round(renewable_target_pct, 2),
            "onsite_gen_allowed": onsite_gen_allowed,
            "ppa_price_usd_mwh": round(ppa_price_usd_mwh, 2),
            "spot_price_avg_usd_mwh": round(spot_price_avg_usd_mwh, 2),
            "battery_cost_usd_kwh": round(battery_cost_usd_kwh, 2),
            # Target labels
            "recommended_mix": recommended_mix,
            "vendor_recommendation": vendor,
            "contract_lead_time_months": contract_lead_time,
            "forecasted_cost_usd_mwh": forecasted_cost
        }
    
    def generate_dataset(
        self, 
        n_samples: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate full synthetic dataset.
        
        Args:
            n_samples: Number of samples (uses config default if None)
            save_path: Path to save CSV (optional)
        
        Returns:
            DataFrame with all samples
        """
        n = n_samples or self.num_samples
        
        samples = [self.generate_single_sample() for _ in range(n)]
        
        # Convert to DataFrame
        df = pd.DataFrame(samples)
        
        # Serialize list/dict columns for CSV storage
        df['gpu_util_profile'] = df['gpu_util_profile'].apply(json.dumps)
        df['power_trace_kw'] = df['power_trace_kw'].apply(json.dumps)
        df['recommended_mix'] = df['recommended_mix'].apply(json.dumps)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Dataset saved to {save_path}")
        
        return df
    
    def load_dataset(self, path: str) -> pd.DataFrame:
        """
        Load dataset from CSV and parse serialized columns.
        """
        df = pd.read_csv(path)
        
        # Parse JSON columns
        df['gpu_util_profile'] = df['gpu_util_profile'].apply(json.loads)
        df['power_trace_kw'] = df['power_trace_kw'].apply(json.loads)
        df['recommended_mix'] = df['recommended_mix'].apply(json.loads)
        
        return df


def main():
    """Generate and save synthetic dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic power procurement dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/synthetic_dataset.csv", help="Output path")
    
    args = parser.parse_args()
    
    generator = PowerDataGenerator(args.config)
    df = generator.generate_dataset(n_samples=args.samples, save_path=args.output)
    
    print(f"\nGenerated {len(df)} samples")
    print(f"\nDataset columns:\n{df.columns.tolist()}")
    print(f"\nSample statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
