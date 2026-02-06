"""
RL Policy Optimizer for Power Procurement
==========================================
Gym environment and RL training using stable-baselines3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import yaml


class PowerProcurementEnv(gym.Env):
    """
    OpenAI Gym environment for power procurement optimization.
    
    State: [demand_curve (48), market_prices (3), grid_limit, renewable_target]
    Action: [ppa_pct, spot_pct, battery_pct] (continuous, sum to 1)
    Reward: Minimize cost + penalties for unmet demand and emissions
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.time_steps = self.config['data']['time_steps']
        
        # State space: demand curve + prices + constraints
        # 48 (demand) + 3 (prices) + 2 (grid_limit, renewable_target)
        state_dim = self.time_steps + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Action space: procurement mix percentages (PPA, Spot, Battery)
        # Using Box for continuous actions
        self.action_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )
        
        # Episode parameters
        self.max_steps = 24  # One day of hourly decisions
        self.current_step = 0
        self.state = None
        
        # Cost parameters
        self.unmet_penalty = 1000  # $/MWh for unmet demand
        self.emission_penalty = 50  # $/ton CO2 equivalent
        
    def _generate_demand_curve(self) -> np.ndarray:
        """Generate a realistic demand curve."""
        t = np.arange(self.time_steps)
        
        # Base diurnal pattern
        base = 50 + 30 * np.sin(2 * np.pi * (t - 6) / 24)
        
        # Random spikes and noise
        noise = np.random.normal(0, 5, self.time_steps)
        spikes = np.random.choice([0, 10, 20], self.time_steps, p=[0.8, 0.15, 0.05])
        
        return np.clip(base + noise + spikes, 10, 150).astype(np.float32)
    
    def _generate_prices(self) -> np.ndarray:
        """Generate market prices [PPA, Spot, Battery cost]."""
        ppa = np.random.uniform(30, 80)
        spot = np.random.uniform(25, 120)
        battery = np.random.uniform(100, 300)
        return np.array([ppa, spot, battery], dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Generate episode parameters
        self.demand_curve = self._generate_demand_curve()
        self.prices = self._generate_prices()
        self.grid_limit = np.random.uniform(50, 200)
        self.renewable_target = np.random.uniform(0.5, 1.0)
        
        # Construct state
        self.state = np.concatenate([
            self.demand_curve / 100,  # Normalize demand
            self.prices / 100,  # Normalize prices
            [self.grid_limit / 100, self.renewable_target]
        ]).astype(np.float32)
        
        return self.state, {}
    
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
        
        # Get current demand
        demand = self.demand_curve[self.current_step % self.time_steps]
        
        # Calculate procurement (in MW)
        ppa_mw = ppa_pct * self.grid_limit
        spot_mw = spot_pct * self.grid_limit
        battery_mw = battery_pct * self.grid_limit * 0.5  # Battery provides peak shaving
        
        total_supply = ppa_mw + spot_mw + battery_mw
        
        # Calculate costs
        ppa_cost = ppa_mw * self.prices[0]
        spot_cost = spot_mw * self.prices[1]
        battery_cost = battery_mw * self.prices[2] * 0.01  # Amortized
        
        base_cost = ppa_cost + spot_cost + battery_cost
        
        # Penalties
        unmet_demand = max(0, demand - total_supply)
        unmet_penalty = unmet_demand * self.unmet_penalty
        
        # Renewable penalty (spot is assumed non-renewable)
        renewable_fraction = ppa_mw / (total_supply + 1e-8)
        renewable_shortfall = max(0, self.renewable_target - renewable_fraction)
        emission_penalty = renewable_shortfall * self.emission_penalty * demand
        
        # Total cost (negative reward)
        total_cost = base_cost + unmet_penalty + emission_penalty
        reward = -total_cost / 1000  # Scale for training stability
        
        # Update step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Update demand for next step (shift curve)
        if not terminated:
            self.demand_curve = np.roll(self.demand_curve, -1)
            self.demand_curve[-1] = self._generate_demand_curve()[-1]
            
            self.state = np.concatenate([
                self.demand_curve / 100,
                self.prices / 100,
                [self.grid_limit / 100, self.renewable_target]
            ]).astype(np.float32)
        
        info = {
            'demand': demand,
            'supply': total_supply,
            'cost': base_cost,
            'unmet': unmet_demand,
            'renewable_fraction': renewable_fraction
        }
        
        return self.state, reward, terminated, False, info
    
    def render(self, mode: str = "human"):
        if mode == "human":
            print(f"Step {self.current_step}: Demand={self.demand_curve[0]:.1f} MW")


class RLOptimizer:
    """High-level RL training interface."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rl_config = self.config['rl']
        self.env = PowerProcurementEnv(config_path)
        self.model = None
    
    def train(self, verbose: bool = True):
        """Train RL policy."""
        try:
            from stable_baselines3 import PPO, DQN
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print("stable-baselines3 not installed. Run: pip install stable-baselines3")
            return None
        
        # Wrap environment
        env = DummyVecEnv([lambda: self.env])
        
        # Create model
        if self.rl_config['algorithm'] == "PPO":
            self.model = PPO(
                "MlpPolicy", env,
                learning_rate=self.rl_config['learning_rate'],
                n_steps=self.rl_config['n_steps'],
                batch_size=self.rl_config['batch_size'],
                gamma=self.rl_config['gamma'],
                verbose=1 if verbose else 0
            )
        else:
            self.model = PPO("MlpPolicy", env, verbose=1 if verbose else 0)
        
        # Train
        print(f"🚀 Training {self.rl_config['algorithm']} for {self.rl_config['total_timesteps']} steps...")
        self.model.learn(total_timesteps=self.rl_config['total_timesteps'])
        
        return self.model
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """Evaluate trained policy."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        rewards = []
        costs = []
        unmet = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_cost = 0
            episode_unmet = 0
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                episode_cost += info['cost']
                episode_unmet += info['unmet']
            
            rewards.append(episode_reward)
            costs.append(episode_cost)
            unmet.append(episode_unmet)
        
        return {
            'mean_reward': np.mean(rewards),
            'mean_cost': np.mean(costs),
            'mean_unmet_demand': np.mean(unmet),
            'std_reward': np.std(rewards)
        }
    
    def save(self, path: str):
        """Save trained model."""
        if self.model:
            self.model.save(f"{path}/rl_policy")
            print(f"RL model saved to {path}/rl_policy.zip")
    
    def load(self, path: str):
        """Load trained model."""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(f"{path}/rl_policy", env=self.env)
            print(f"RL model loaded from {path}/rl_policy.zip")
        except ImportError:
            print("stable-baselines3 not installed.")


def main():
    """Train and evaluate RL optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL optimizer")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--save", type=str, default="models/rl")
    
    args = parser.parse_args()
    
    optimizer = RLOptimizer(args.config)
    optimizer.train()
    
    print("\n📊 Evaluation:")
    metrics = optimizer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
    
    optimizer.save(args.save)


if __name__ == "__main__":
    main()
