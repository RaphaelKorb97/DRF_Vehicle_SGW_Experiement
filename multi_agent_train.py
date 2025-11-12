"""Multi-Agent Training Script - trains each vehicle with its own RL policy"""
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from multi_agent_environment import MultiAgentACCEnvironment, SingleAgentWrapper
from config import *
import gymnasium as gym

def get_device():
    """Get the best available device for training"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU acceleration")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device

DEVICE = get_device()

class PerturbationWrapper(gym.Wrapper):
    """Wrapper that adds random perturbations during training for robustness"""
    
    def __init__(self, env, perturbation_probability=0.03):
        super().__init__(env)
        self.env = env
        self.perturbation_probability = perturbation_probability
        self.perturbation_active = False
        self.perturbation_duration = 0
        self.perturbation_max_duration = 30  # 3 seconds
        self.perturbation_target = None
        
    def reset(self, **kwargs):
        self.perturbation_active = False
        self.perturbation_duration = 0
        self.perturbation_target = None
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # Randomly trigger perturbations
        if not self.perturbation_active and np.random.random() < self.perturbation_probability:
            if hasattr(self.env, 'multi_env') and hasattr(self.env.multi_env, 'vehicles'):
                num_vehicles = len(self.env.multi_env.vehicles)
                if num_vehicles > 1:
                    self.perturbation_target = np.random.randint(0, num_vehicles)
                    self.perturbation_active = True
                    self.perturbation_duration = 0
        
        # Apply perturbation
        if self.perturbation_active:
            if self.perturbation_duration < self.perturbation_max_duration:
                if hasattr(self.env, 'multi_env') and self.perturbation_target == self.env.agent_id:
                    action = np.array([MAX_DECELERATION * 0.8])
                self.perturbation_duration += 1
            else:
                self.perturbation_active = False
        
        return self.env.step(action)

def create_training_env(agent_id, perturbations=True):
    """Create environment with perturbation training"""
    multi_env = MultiAgentACCEnvironment()
    single_env = SingleAgentWrapper(multi_env, agent_id)
    
    # Add sensor noise
    single_env.noise_enabled = True
    single_env.noise_std = 0.5
    
    # Add perturbations if enabled
    if perturbations:
        single_env = PerturbationWrapper(single_env, perturbation_probability=0.03)
    
    return single_env

def train_agent(agent_id, total_timesteps=TOTAL_TIMESTEPS):
    """Train a single agent using PPO"""
    print(f"\n{'='*40}")
    print(f"Training Agent {agent_id}")
    print(f"{'='*40}")
    print(f"Device: {DEVICE}")
    
    # Create training environment
    def make_env():
        return create_training_env(agent_id, perturbations=True)
    
    env = make_vec_env(make_env, n_envs=1)
    
    # Create evaluation environment (without perturbations)
    eval_env = Monitor(create_training_env(agent_id, perturbations=False))
    
    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/agent_{agent_id}/",
        log_path=f"logs/agent_{agent_id}/",
        eval_freq=2048,
        deterministic=True,
        render=False
    )
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={'net_arch': [64, 64, 64]},  # 3 hidden layers as per paper
        tensorboard_log=f"logs/agent_{agent_id}",
        verbose=1,
        device=DEVICE
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"models/agent_{agent_id}/final_model")
    print(f"\nAgent {agent_id} training completed!")
    
    env.close()
    eval_env.close()
    
    return model

def train_all_agents():
    """Train all agents sequentially"""
    print("\n" + "="*60)
    print("Multi-Agent Car-Following Training")
    print("="*60)
    print(f"Number of vehicles: {NUM_VEHICLES}")
    print(f"Training timesteps per agent: {TOTAL_TIMESTEPS}")
    print(f"Device: {DEVICE}")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    models = {}
    
    for agent_id in range(NUM_VEHICLES):
        # Create agent-specific directories
        os.makedirs(f"models/agent_{agent_id}", exist_ok=True)
        os.makedirs(f"logs/agent_{agent_id}", exist_ok=True)
        
        # Train this agent
        model = train_agent(agent_id)
        models[agent_id] = model
    
    print("\n" + "="*60)
    print("All agents trained successfully!")
    print("="*60)
    
    return models

def evaluate_agents(num_episodes=5):
    """Evaluate trained agents"""
    print("\nEvaluating trained agents...")
    
    # Create environment
    env = MultiAgentACCEnvironment()
    
    # Load models
    models = {}
    for agent_id in range(NUM_VEHICLES):
        try:
            model_path = f"models/agent_{agent_id}/best_model"
            models[agent_id] = PPO.load(model_path)
            print(f"Loaded model for Agent {agent_id}")
        except:
            print(f"Could not load model for Agent {agent_id}")
            models[agent_id] = None
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        observations, info = env.reset()
        episode_reward = np.zeros(NUM_VEHICLES)
        done = False
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done and step_count < 1000:
            # Get actions from all agents
            actions = []
            for agent_id in range(NUM_VEHICLES):
                if models[agent_id] is not None:
                    action, _ = models[agent_id].predict(observations[agent_id], deterministic=True)
                else:
                    action = env.single_action_space.sample()
                actions.append(action)
            
            # Take step
            observations, rewards, terminated, truncated, info = env.step(actions)
            episode_reward += rewards
            step_count += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        
        print(f"  Total System Reward: {np.sum(episode_reward):.2f}")
        print(f"  Steps: {step_count}")
    
    env.close()
    
    # Print summary
    episode_rewards = np.array(episode_rewards)
    mean_rewards = np.mean(episode_rewards, axis=0)
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("-"*60)
    print(f"Mean system reward: {np.mean(mean_rewards):.2f}")
    print(f"Std system reward: {np.std(mean_rewards):.2f}")
    print("="*60)
    
    return episode_rewards

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Agent Car-Following Training')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                       help='Mode: train or eval')
    parser.add_argument('--agent_id', type=int, default=None,
                       help='Train specific agent (default: train all)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if args.agent_id is not None:
            train_agent(args.agent_id)
        else:
            train_all_agents()
    elif args.mode == 'eval':
        evaluate_agents(num_episodes=args.episodes)
