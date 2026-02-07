"""
Real Data Processor for Power Procurement Model
=================================================

Transforms job_table.parquet into training-ready format for the
power forecasting and procurement recommendation models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import interpolate
import json
import warnings
warnings.filterwarnings('ignore')


class RealDataProcessor:
    """
    Processes real HPC job telemetry data into model-ready format.
    
    Handles:
    - Power consumption time-series resampling
    - Feature extraction from job metadata
    - Unit conversions and normalization
    - Procurement target generation based on actual usage
    """
    
    def __init__(self, target_time_steps: int = 48):
        """
        Args:
            target_time_steps: Number of time steps to resample power traces to
        """
        self.target_time_steps = target_time_steps
        self.power_unit_factor = 1.0  # Adjust if data is in W (divide by 1000 for kW)
        
    def load_parquet(self, path: str) -> pd.DataFrame:
        """Load parquet file."""
        print(f"📂 Loading data from {path}...")
        df = pd.read_parquet(path)
        print(f"   Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def _resample_time_series(
        self, 
        series: List, 
        target_length: int
    ) -> np.ndarray:
        """
        Resample a time series to a fixed length using interpolation.
        
        Args:
            series: Original time series (variable length)
            target_length: Desired output length
            
        Returns:
            Resampled array of target_length
        """
        if series is None or len(series) == 0:
            return np.zeros(target_length)
        
        series = np.array(series, dtype=np.float64)
        
        # Handle single value or very short series
        if len(series) < 2:
            return np.full(target_length, series[0] if len(series) > 0 else 0)
        
        # Original time points (normalized to 0-1)
        x_original = np.linspace(0, 1, len(series))
        
        # Target time points
        x_target = np.linspace(0, 1, target_length)
        
        # Interpolate
        try:
            f = interpolate.interp1d(x_original, series, kind='linear', fill_value='extrapolate')
            resampled = f(x_target)
        except Exception:
            # Fallback to simple resampling
            indices = np.linspace(0, len(series) - 1, target_length).astype(int)
            resampled = series[indices]
        
        return resampled
    
    def _safe_parse_list(self, value) -> List:
        """Safely parse list-like values from DataFrame."""
        if value is None:
            return []
        if isinstance(value, (list, np.ndarray)):
            return list(value)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return []
        return []
    
    def _compute_total_power(
        self,
        node_power: List,
        cpu_power: List,
        mem_power: List
    ) -> np.ndarray:
        """
        Compute total power consumption from components.
        
        The node_power typically includes CPU + GPU + overhead.
        We can use component breakdown for analysis.
        """
        # Use node_power as primary (it's typically the most comprehensive)
        if node_power and len(node_power) > 0:
            return np.array(node_power, dtype=np.float64)
        
        # Fallback: sum components if node_power unavailable
        cpu = np.array(cpu_power if cpu_power else [0], dtype=np.float64)
        mem = np.array(mem_power if mem_power else [0], dtype=np.float64)
        
        # Align lengths
        max_len = max(len(cpu), len(mem))
        if len(cpu) < max_len:
            cpu = np.pad(cpu, (0, max_len - len(cpu)), mode='edge')
        if len(mem) < max_len:
            mem = np.pad(mem, (0, max_len - len(mem)), mode='edge')
        
        return cpu + mem
    
    def _generate_utilization_profile(
        self,
        power_trace: np.ndarray,
        num_gpus: int
    ) -> np.ndarray:
        """
        Estimate GPU utilization from power consumption.
        
        Uses power normalized to estimated TDP as proxy for utilization.
        """
        if num_gpus <= 0:
            # No GPUs - use CPU utilization proxy
            normalized = power_trace / (power_trace.max() + 1e-8)
            return np.clip(normalized, 0.1, 0.99)
        
        # Estimate per-GPU power (assuming ~400W TDP average)
        estimated_tdp = 400 * num_gpus  # Total TDP in Watts
        
        # Normalize power to get utilization estimate
        utilization = power_trace / (estimated_tdp + 1e-8)
        
        # Clip to valid range and add noise for realism
        utilization = np.clip(utilization, 0.1, 0.99)
        
        return utilization
    
    def _estimate_procurement_targets(
        self,
        avg_power_kw: float,
        peak_power_kw: float,
        num_gpus: int,
        run_time_hours: float
    ) -> Dict:
        """
        Estimate procurement targets based on power characteristics.
        
        This creates labeled training data by applying heuristics
        that approximate optimal procurement strategies.
        """
        # Convert to MW for procurement
        avg_power_mw = avg_power_kw / 1000
        peak_power_mw = peak_power_kw / 1000
        
        # Heuristic: larger, longer jobs favor PPA
        stability_score = min(1.0, run_time_hours / 24)  # Longer = more stable
        scale_score = min(1.0, num_gpus / 64)  # Larger = favor PPA
        
        # Base renewable target (industry moving to 80-100%)
        renewable_target = np.random.uniform(0.7, 1.0)
        
        # PPA covers baseload + renewable requirements
        ppa_fraction = 0.5 + 0.3 * stability_score + 0.1 * scale_score
        ppa_fraction = min(ppa_fraction, renewable_target)
        
        # Spot covers variable demand
        spot_fraction = 1.0 - ppa_fraction - 0.05  # Leave room for battery
        
        # Battery for peak shaving
        peak_ratio = peak_power_mw / (avg_power_mw + 1e-8)
        battery_fraction = 0.05 + 0.1 * min(1.0, (peak_ratio - 1.0))
        
        # Normalize
        total = ppa_fraction + spot_fraction + battery_fraction
        
        # Calculate actual MW values
        ppa_mw = avg_power_mw * (ppa_fraction / total)
        spot_mw = avg_power_mw * (spot_fraction / total)
        battery_mw = avg_power_mw * (battery_fraction / total)
        
        # Market prices (simulated realistic ranges)
        ppa_price = np.random.uniform(35, 75)
        spot_price = np.random.uniform(30, 110)
        battery_cost = np.random.uniform(120, 280)
        
        # Cost estimate
        cost_usd_mwh = (ppa_mw * ppa_price + spot_mw * spot_price) / (ppa_mw + spot_mw + 1e-8)
        
        # Lead time based on PPA fraction and scale
        base_lead = 6
        lead_time = int(base_lead + ppa_fraction * 12 + scale_score * 6)
        lead_time = min(lead_time, 36)
        
        return {
            'recommended_mix': {
                'ppa': round(ppa_mw, 4),
                'spot': round(spot_mw, 4),
                'battery': round(battery_mw, 4)
            },
            'renewable_target_pct': round(renewable_target * 100, 1),
            'ppa_price_usd_mwh': round(ppa_price, 2),
            'spot_price_avg_usd_mwh': round(spot_price, 2),
            'battery_cost_usd_kwh': round(battery_cost, 2),
            'forecasted_cost_usd_mwh': round(cost_usd_mwh, 2),
            'contract_lead_time_months': lead_time
        }
    
    def process_record(self, row: pd.Series) -> Optional[Dict]:
        """
        Process a single job record into model-ready format.
        
        Args:
            row: DataFrame row representing a job
            
        Returns:
            Processed record dict or None if invalid
        """
        try:
            # Parse power consumption arrays
            node_power = self._safe_parse_list(row.get('node_power_consumption'))
            cpu_power = self._safe_parse_list(row.get('cpu_power_consumption'))
            mem_power = self._safe_parse_list(row.get('mem_power_consumption'))
            
            # Compute total power
            total_power = self._compute_total_power(node_power, cpu_power, mem_power)
            
            if len(total_power) < 5:  # Too short to be useful
                return None
            
            # Resample to target length
            power_trace = self._resample_time_series(total_power, self.target_time_steps)
            
            # Convert to kW (assuming input is in Watts)
            power_trace_kw = power_trace * self.power_unit_factor / 1000
            
            # Get GPU info
            num_gpus = int(row.get('num_gpus_alloc', 0) or 0)
            num_nodes = int(row.get('num_nodes_alloc', 1) or 1)
            num_cores = int(row.get('num_cores_alloc', 1) or 1)
            
            # Calculate run time in hours
            run_time_minutes = float(row.get('run_time', 60) or 60)
            run_time_hours = run_time_minutes / 60
            
            # Generate utilization profile
            gpu_util_profile = self._generate_utilization_profile(power_trace, num_gpus)
            
            # Calculate power statistics
            avg_power_kw = float(np.mean(power_trace_kw))
            peak_power_kw = float(np.max(power_trace_kw))
            
            if avg_power_kw <= 0:
                return None
            
            # Estimate rack density (power per node)
            rack_density_kw = avg_power_kw / max(1, num_nodes)
            
            # Estimate grid capacity (with headroom)
            grid_capacity_mw = peak_power_kw / 1000 * 2.0
            
            # Infer workload type from characteristics
            util_variance = np.var(gpu_util_profile)
            if util_variance < 0.02 and num_gpus > 8:
                workload_type = "LLM_Training"
            else:
                workload_type = "Inference"
            
            # Estimate procurement targets
            targets = self._estimate_procurement_targets(
                avg_power_kw, peak_power_kw, num_gpus, run_time_hours
            )
            
            # Infer GPU type based on power per GPU
            if num_gpus > 0:
                power_per_gpu = avg_power_kw / num_gpus
                if power_per_gpu > 0.6:
                    gpu_type = "H100"
                elif power_per_gpu > 0.35:
                    gpu_type = "A100"
                else:
                    gpu_type = "MI300X"
            else:
                gpu_type = "A100"  # Default
            
            # Build output record
            record = {
                # Hardware specs
                'num_gpus': num_gpus,
                'gpu_type': gpu_type,
                'rack_density_kw': round(rack_density_kw, 2),
                'num_nodes': num_nodes,
                'num_cores': num_cores,
                
                # Workload
                'workload_type': workload_type,
                'batch_size': 32,  # Default (not in source data)
                'seq_length': 2048,  # Default
                'fp_precision': 'bf16',  # Default
                'run_time_hours': round(run_time_hours, 2),
                
                # Time series
                'gpu_util_profile': gpu_util_profile.tolist(),
                'power_trace_kw': power_trace_kw.tolist(),
                
                # Power statistics
                'burst_peak_kw': round(peak_power_kw, 2),
                'avg_power_kw': round(avg_power_kw, 2),
                
                # Site constraints
                'grid_capacity_mw': round(grid_capacity_mw, 2),
                'onsite_gen_allowed': False,
                
                # Market prices and targets (from estimation)
                **targets,
                
                # Metadata from original
                'job_id': int(row.get('job_id', 0)),
                'job_state': str(row.get('job_state', 'UNKNOWN')),
                'user_id': int(row.get('user_id', 0) or 0),
            }
            
            return record
            
        except Exception as e:
            # Skip problematic records
            return None
    
    def process_dataset(
        self,
        df: pd.DataFrame,
        filter_states: Optional[List[str]] = None,
        min_gpus: int = 0,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process entire dataset into model-ready format.
        
        Args:
            df: Raw DataFrame from parquet
            filter_states: Job states to include (e.g., ['COMPLETED'])
            min_gpus: Minimum GPUs to include job
            max_samples: Maximum samples to process
            
        Returns:
            Processed DataFrame ready for training
        """
        print(f"\n🔄 Processing {len(df)} records...")
        
        # Filter by job state if specified
        if filter_states:
            df = df[df['job_state'].isin(filter_states)]
            print(f"   After state filter: {len(df)} records")
        
        # Filter by GPU count
        if min_gpus > 0:
            df = df[df['num_gpus_alloc'] >= min_gpus]
            print(f"   After GPU filter (>={min_gpus}): {len(df)} records")
        
        # Limit samples if specified
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            print(f"   Sampled to: {len(df)} records")
        
        # Process each record
        processed_records = []
        failed_count = 0
        
        for idx, row in df.iterrows():
            record = self.process_record(row)
            if record:
                processed_records.append(record)
            else:
                failed_count += 1
        
        print(f"   ✅ Successfully processed: {len(processed_records)}")
        print(f"   ⚠️  Skipped (invalid data): {failed_count}")
        
        # Convert to DataFrame
        result_df = pd.DataFrame(processed_records)
        
        # Serialize list columns for CSV storage
        result_df['gpu_util_profile'] = result_df['gpu_util_profile'].apply(json.dumps)
        result_df['power_trace_kw'] = result_df['power_trace_kw'].apply(json.dumps)
        result_df['recommended_mix'] = result_df['recommended_mix'].apply(json.dumps)
        
        return result_df
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        output_path: str
    ):
        """Save processed DataFrame to CSV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved processed data to {output_path}")
        print(f"   Shape: {df.shape}")
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate dataset statistics."""
        stats = {
            'total_records': len(df),
            'gpu_distribution': df['num_gpus'].describe().to_dict(),
            'workload_types': df['workload_type'].value_counts().to_dict(),
            'gpu_types': df['gpu_type'].value_counts().to_dict(),
            'avg_power_kw': df['avg_power_kw'].describe().to_dict(),
            'peak_power_kw': df['burst_peak_kw'].describe().to_dict(),
        }
        return stats


def main():
    """Main processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process job telemetry data")
    parser.add_argument("--input", type=str, default="data/job_table.parquet",
                       help="Input parquet file path")
    parser.add_argument("--output", type=str, default="data/processed_dataset.csv",
                       help="Output CSV file path")
    parser.add_argument("--filter-states", type=str, nargs='+',
                       default=['COMPLETED', 'CANCELLED', 'FAILED'],
                       help="Job states to include")
    parser.add_argument("--min-gpus", type=int, default=0,
                       help="Minimum GPUs required")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to process")
    parser.add_argument("--time-steps", type=int, default=48,
                       help="Target time series length")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = RealDataProcessor(target_time_steps=args.time_steps)
    
    # Load data
    df = processor.load_parquet(args.input)
    
    # Process
    processed_df = processor.process_dataset(
        df,
        filter_states=args.filter_states,
        min_gpus=args.min_gpus,
        max_samples=args.max_samples
    )
    
    # Save
    processor.save_processed_data(processed_df, args.output)
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    stats = processor.get_statistics(processed_df)
    print(f"   Total Records: {stats['total_records']}")
    print(f"   Workload Types: {stats['workload_types']}")
    print(f"   GPU Types: {stats['gpu_types']}")
    print(f"   Avg Power (kW): mean={stats['avg_power_kw']['mean']:.2f}, "
          f"std={stats['avg_power_kw']['std']:.2f}")
    print(f"   Peak Power (kW): mean={stats['peak_power_kw']['mean']:.2f}, "
          f"max={stats['peak_power_kw']['max']:.2f}")


if __name__ == "__main__":
    main()
