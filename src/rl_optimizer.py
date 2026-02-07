"""
RL Policy Optimizer for Power Procurement
==========================================
Gym environment and RL training using stable-baselines3.
Enhanced to learn from real HPC workload patterns.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
import yaml
import json
from pathlib import Path


class PowerProcurementEnv(gym.Env):
    """
    OpenAI Gym environment for power procurement optimization.
    
    State: [demand_curve (48), market_prices (3), grid_limit, renewable_target, hour_of_day]
    Action: [ppa_pct, spot_pct, battery_pct] (continuous, sum to 1)
    Reward: Minimize cost + penalties for unmet demand and emissions
    
    Enhanced to use real demand patterns from HPC job data.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self, 
        config_path: str = "config.yaml",
        real_data_path: Optional[str] = None
    ):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.time_steps = self.config['data']['time_steps']
        
        # Load real demand patterns if available
        self.real_patterns = []
        if real_data_path and Path(real_data_path).exists():
            self._load_real_patterns(real_data_path)
        
        # State space: demand curve + prices + constraints + hour
        # 48 (demand) + 3 (prices) + 2 (grid_limit, renewable_target) + 1 (hour)
        state_dim = self.time_steps + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Action space: procurement mix percentages (PPA, Spot, Battery)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )
        
        # Episode parameters
        self.max_steps = 48  # Two days of hourly decisions
        self.current_step = 0
        self.state = None
        
        # Cost parameters (realistic market rates)
        self.unmet_penalty = 500  # $/MWh for unmet demand
        self.emission_penalty = 75  # $/ton CO2 equivalent
        self.battery_efficiency = 0.85  # Round-trip efficiency
        
        # Battery state
        self.battery_soc = 0.5  # State of charge (0-1)
        self.battery_capacity = 10.0  # MWh
        
    def _load_real_patterns(self, path: str):
        """Load real demand patterns from processed data."""
        try:
            df = pd.read_csv(path, nrows=5000)  # Load subset for memory
            
            for _, row in df.iterrows():
                power_trace = json.loads(row['power_trace_kw'])
                if len(power_trace) == self.time_steps:
                    # Convert kW to MW
                    self.real_patterns.append(np.array(power_trace) / 1000)
            
            print(f"📂 Loaded {len(self.real_patterns)} real demand patterns")
            
        except Exception as e:
            print(f"⚠️ Could not load real patterns: {e}")
    
    def _generate_demand_curve(self) -> np.ndarray:
        """Generate a demand curve (real or synthetic)."""
        
        # Use real pattern if available
        if self.real_patterns and np.random.random() > 0.3:
            idx = np.random.randint(len(self.real_patterns))
            base = self.real_patterns[idx].copy()
            # Add small noise
            noise = np.random.normal(0, 0.02, self.time_steps)
            return np.clip(base + noise, 0.01, 500).astype(np.float32)
        
        # Synthetic pattern
        t = np.arange(self.time_steps)
        
        # Base diurnal pattern (data center workload)
        base = 5 + 3 * np.sin(2 * np.pi * (t - 6) / 24)
        
        # Random variations based on workload type
        workload_type = np.random.choice(['training', 'inference', 'mixed'])
        
        if workload_type == 'training':
            # Sustained high load
            base = base * 1.5 + np.random.uniform(0, 2)
        elif workload_type == 'inference':
            # Variable with spikes
            spikes = np.random.choice([0, 2, 5], self.time_steps, p=[0.6, 0.3, 0.1])
            base = base + spikes
        else:
            # Mixed workload
            base = base * np.random.uniform(0.8, 1.2, self.time_steps)
        
        noise = np.random.normal(0, 0.5, self.time_steps)
        
        return np.clip(base + noise, 0.5, 50).astype(np.float32)
    
    def _generate_prices(self) -> np.ndarray:
        """Generate market prices [PPA, Spot, Battery cost]."""
        # PPA: stable long-term contract
        ppa = np.random.uniform(35, 75)
        
        # Spot: volatile, can be cheaper or more expensive
        spot_base = np.random.uniform(30, 90)
        
        # Battery storage cost ($/kWh installed, amortized)
        battery = np.random.uniform(120, 280)
        
        return np.array([ppa, spot_base, battery], dtype=np.float32)
    
    def _get_spot_price_variation(self) -> float:
        """Get hourly spot price variation based on time of day."""
        hour = self.current_step % 24
        
        # Peak hours (6pm-10pm) have higher prices
        if 18 <= hour <= 22:
            return np.random.uniform(1.2, 1.8)
        # Off-peak (midnight to 6am) have lower prices
        elif 0 <= hour <= 6:
            return np.random.uniform(0.6, 0.9)
        else:
            return np.random.uniform(0.9, 1.1)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.battery_soc = np.random.uniform(0.3, 0.7)  # Random initial charge
        
        # Generate episode parameters
        self.demand_curve = self._generate_demand_curve()
        self.base_prices = self._generate_prices()
        self.grid_limit = np.random.uniform(10, 100)  # MW
        self.renewable_target = np.random.uniform(0.6, 1.0)
        
        # Current prices (with spot variation)
        self.prices = self.base_prices.copy()
        self.prices[1] *= self._get_spot_price_variation()
        
        # Construct state
        self.state = self._get_state()
        
        return self.state, {}
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        # Normalize demand (based on grid limit)
        norm_demand = self.demand_curve / (self.grid_limit + 1e-8)
        
        # Normalize prices
        norm_prices = self.prices / 100
        
        # Hour encoding (cyclical)
        hour = self.current_step % 24
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        return np.concatenate([
            norm_demand,
            norm_prices,
            [self.grid_limit / 100, self.renewable_target],
            [self.battery_soc]
        ]).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of procurement decision.
        
        Args:
            action: [ppa_pct, spot_pct, battery_pct]
        
        Returns:
            state, reward, terminated, truncated, info
        """
        # Normalize action to sum to 1
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        ppa_pct, spot_pct, battery_pct = action
        
        # Get current demand (MW)
        demand = self.demand_curve[self.current_step % self.time_steps]
        
        # Update spot price with hourly variation
        current_spot = self.base_prices[1] * self._get_spot_price_variation()
        
        # Calculate procurement (in MW)
        ppa_mw = ppa_pct * self.grid_limit
        spot_mw = spot_pct * self.grid_limit
        
        # Battery can charge or discharge
        battery_action = battery_pct * 2 - 1  # Convert to -1 to 1
        battery_mw = battery_action * self.battery_capacity * 0.25  # Max 25% per step
        
        # Update battery SOC
        if battery_mw > 0:  # Discharging
            actual_battery = min(battery_mw, self.battery_soc * self.battery_capacity)
            actual_battery *= self.battery_efficiency
            self.battery_soc -= actual_battery / self.battery_capacity
        else:  # Charging
            charging_power = -battery_mw
            actual_battery = -charging_power
            self.battery_soc += charging_power * self.battery_efficiency / self.battery_capacity
        
        self.battery_soc = np.clip(self.battery_soc, 0, 1)
        
        # Total supply
        total_supply = ppa_mw + spot_mw + max(0, actual_battery)
        
        # Calculate costs ($/hour)
        ppa_cost = ppa_mw * self.prices[0] / 1000  # Convert to $/kWh equivalent
        spot_cost = spot_mw * current_spot / 1000
        
        # Battery degradation cost
        battery_cost = abs(battery_mw) * self.prices[2] * 0.001  # Small degradation
        
        base_cost = ppa_cost + spot_cost + battery_cost
        
        # Penalties
        unmet_demand = max(0, demand - total_supply)
        unmet_penalty = unmet_demand * self.unmet_penalty / 1000
        
        # Renewable penalty (PPA assumed renewable, spot assumed fossil)
        renewable_supply = ppa_mw + max(0, actual_battery)
        renewable_fraction = renewable_supply / (total_supply + 1e-8)
        renewable_shortfall = max(0, self.renewable_target - renewable_fraction)
        emission_penalty = renewable_shortfall * self.emission_penalty * demand / 1000
        
        # Oversupply penalty (wasted energy)
        oversupply = max(0, total_supply - demand)
        oversupply_penalty = oversupply * 0.1  # Small penalty for waste
        
        # Total cost (negative reward)
        total_cost = base_cost + unmet_penalty + emission_penalty + oversupply_penalty
        
        # Reward shaping
        # Bonus for meeting demand efficiently
        efficiency = min(demand, total_supply) / (demand + 1e-8)
        efficiency_bonus = efficiency * 0.5
        
        # Bonus for meeting renewable target
        renewable_bonus = min(renewable_fraction / self.renewable_target, 1.0) * 0.3
        
        reward = -total_cost + efficiency_bonus + renewable_bonus
        
        # Update step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Update state for next step
        if not terminated:
            # Shift demand curve and add new value
            self.demand_curve = np.roll(self.demand_curve, -1)
            
            # Generate new demand based on pattern
            if self.real_patterns and np.random.random() > 0.5:
                idx = np.random.randint(len(self.real_patterns))
                self.demand_curve[-1] = self.real_patterns[idx][-1]
            else:
                self.demand_curve[-1] = self._generate_demand_curve()[-1]
            
            self.state = self._get_state()
        
        info = {
            'demand': float(demand),
            'supply': float(total_supply),
            'cost': float(base_cost),
            'unmet': float(unmet_demand),
            'renewable_fraction': float(renewable_fraction),
            'battery_soc': float(self.battery_soc),
            'spot_price': float(current_spot)
        }
        
        return self.state, float(reward), terminated, False, info
    
    def render(self, mode: str = "human"):
        if mode == "human":
            demand = self.demand_curve[0]
            print(f"Step {self.current_step:2d}: Demand={demand:.2f} MW, "
                  f"Battery={self.battery_soc*100:.0f}%, "
                  f"Prices=[PPA:{self.prices[0]:.0f}, Spot:{self.prices[1]:.0f}]")


class RLOptimizer:
    """High-level RL training interface with real data support."""
    
    def __init__(
        self, 
        config_path: str = "config.yaml",
        data_path: Optional[str] = "data/processed_dataset.csv"
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rl_config = self.config['rl']
        self.env = PowerProcurementEnv(config_path, data_path)
        self.model = None
        self.training_history = []
    
    def train(self, verbose: bool = True):
        """Train RL policy."""
        try:
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from stable_baselines3.common.callbacks import EvalCallback
        except ImportError:
            print("⚠️ stable-baselines3 not installed. Run: pip install stable-baselines3")
            return None
        
        # Wrap environment
        def make_env():
            return PowerProcurementEnv(
                self.config.get('config_path', 'config.yaml'),
                'data/processed_dataset.csv'
            )
        
        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        # Create eval environment
        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        # Create model
        algorithm = self.rl_config.get('algorithm', 'PPO')
        
        if algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy", 
                env,
                learning_rate=self.rl_config.get('learning_rate', 3e-4),
                n_steps=self.rl_config.get('n_steps', 2048),
                batch_size=self.rl_config.get('batch_size', 64),
                n_epochs=10,
                gamma=self.rl_config.get('gamma', 0.99),
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1 if verbose else 0,
                tensorboard_log="./logs/rl_tensorboard/"
            )
        elif algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=self.rl_config.get('learning_rate', 3e-4),
                batch_size=self.rl_config.get('batch_size', 256),
                gamma=self.rl_config.get('gamma', 0.99),
                verbose=1 if verbose else 0
            )
        else:
            self.model = PPO("MlpPolicy", env, verbose=1 if verbose else 0)
        
        # Setup callback
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path='./models/rl/best/',
            log_path='./logs/rl/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Train
        total_timesteps = self.rl_config.get('total_timesteps', 100000)
        print(f"🚀 Training {algorithm} for {total_timesteps:,} steps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        return self.model
    
    def evaluate(self, n_episodes: int = 20) -> Dict:
        """Evaluate trained policy."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        rewards = []
        costs = []
        unmet = []
        renewable_fractions = []
        
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_cost = 0
            episode_unmet = 0
            episode_renewable = []
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                episode_cost += info['cost']
                episode_unmet += info['unmet']
                episode_renewable.append(info['renewable_fraction'])
            
            rewards.append(episode_reward)
            costs.append(episode_cost)
            unmet.append(episode_unmet)
            renewable_fractions.append(np.mean(episode_renewable))
        
        metrics = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_cost': np.mean(costs),
            'mean_unmet_demand': np.mean(unmet),
            'mean_renewable_fraction': np.mean(renewable_fractions),
            'demand_satisfaction': 1 - (np.mean(unmet) / 10)  # Approximate
        }
        
        return metrics
    
    def get_action(self, state: Dict) -> Dict:
        """Get action for given state (for inference)."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Construct state vector
        demand = np.array(state.get('demand_curve', np.zeros(48)))
        prices = np.array([
            state.get('ppa_price', 50),
            state.get('spot_price', 70),
            state.get('battery_cost', 180)
        ])
        
        obs = np.concatenate([
            demand / 100,
            prices / 100,
            [state.get('grid_limit', 100) / 100, state.get('renewable_target', 0.8)],
            [state.get('battery_soc', 0.5)]
        ]).astype(np.float32)
        
        action, _ = self.model.predict(obs, deterministic=True)
        action = action / (action.sum() + 1e-8)
        
        return {
            'ppa_pct': float(action[0]),
            'spot_pct': float(action[1]),
            'battery_pct': float(action[2])
        }
    
    def save(self, path: str):
        """Save trained model."""
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save(f"{path}/rl_policy")
            print(f"✅ RL model saved to {path}/rl_policy.zip")
    
    def load(self, path: str):
        """Load trained model."""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(f"{path}/rl_policy", env=self.env)
            print(f"✅ RL model loaded from {path}/rl_policy.zip")
        except ImportError:
            print("⚠️ stable-baselines3 not installed.")
        except Exception as e:
            print(f"⚠️ Error loading RL model: {e}")


def main():
    """Train and evaluate RL optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL optimizer")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data", type=str, default="data/processed_dataset.csv")
    parser.add_argument("--save", type=str, default="models/rl")
    parser.add_argument("--timesteps", type=int, default=100000)
    
    args = parser.parse_args()
    
    optimizer = RLOptimizer(args.config, args.data)
    optimizer.train()
    
    print("\n📊 Evaluation:")
    metrics = optimizer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    optimizer.save(args.save)


if __name__ == "__main__":
    main()

