"""Model Evaluation Script - tests trained models with perturbations"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_agent_environment import MultiAgentACCEnvironment
from config import *
import os
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Evaluator for testing trained models"""
    
    def __init__(self):
        self.results = []
        self.models_cache = {}
        self.observation_history = {}
        self.obs_history_length = 3
        
    def load_model(self, model_path):
        """Load a PPO model"""
        try:
            if model_path in self.models_cache:
                return self.models_cache[model_path]
            
            model = PPO.load(model_path)
            self.models_cache[model_path] = model
            return model
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            return None
    
    def create_enhanced_observation(self, agent_id, base_obs):
        """Create enhanced 16D observation from 4D base observation"""
        if agent_id not in self.observation_history:
            self.observation_history[agent_id] = []
        
        # Store observation history
        self.observation_history[agent_id].append(base_obs.copy())
        
        # Keep only recent history
        if len(self.observation_history[agent_id]) > self.obs_history_length:
            self.observation_history[agent_id].pop(0)
        
        # Pad history if needed
        while len(self.observation_history[agent_id]) < self.obs_history_length:
            self.observation_history[agent_id].insert(0, base_obs.copy())
        
        # Flatten observation history
        obs_history_flat = np.concatenate(self.observation_history[agent_id])
        
        # Extract features
        ego_velocity = base_obs[0]
        front_distance = base_obs[1]
        relative_velocity = base_obs[2]
        
        # Calculate additional features
        time_to_collision = front_distance / max(relative_velocity, 0.1) if relative_velocity > 0 else 100.0
        desired_distance = ego_velocity * DESIRED_TIME_HEADWAY + MIN_DISTANCE
        distance_error = front_distance - desired_distance
        velocity_efficiency = 1.0 - abs(ego_velocity - DESIRED_VELOCITY) / DESIRED_VELOCITY
        
        additional_features = np.array([
            time_to_collision / 10.0,
            distance_error / 50.0,
            velocity_efficiency,
            0.0
        ], dtype=np.float32)
        
        # Combine to 16 dimensions
        enhanced_obs = np.concatenate([obs_history_flat, additional_features])
        
        return enhanced_obs
    
    def run_evaluation_episode(self, env, models, max_steps=5000, perturbation_prob=0.005):
        """Run a single evaluation episode"""
        observations, info = env.reset()
        
        # Clear observation history for new episode
        self.observation_history = {}
        
        episode_data = {
            'velocities': [[] for _ in range(env.num_agents)],
            'distances': [],
            'collisions': 0,
            'total_reward': 0.0,
            'step_count': 0,
            'perturbations_triggered': 0
        }
        
        step_count = 0
        perturbation_active = False
        perturbation_step = 0
        perturbation_duration = 30
        
        while step_count < max_steps:
            # Get actions from models
            actions = []
            for agent_id in range(env.num_agents):
                if agent_id in models and models[agent_id] is not None:
                    try:
                        base_obs = observations[agent_id]
                        # Create enhanced observation (16D) for models trained with enhanced obs
                        enhanced_obs = self.create_enhanced_observation(agent_id, base_obs)
                        action, _ = models[agent_id].predict(enhanced_obs, deterministic=True)
                    except Exception as e:
                        action = np.array([0.0])
                else:
                    # Fallback safe action
                    obs = observations[agent_id]
                    if obs[1] < 20.0:
                        action = np.array([-2.0])
                    elif obs[0] < DESIRED_VELOCITY * 0.9:
                        action = np.array([1.0])
                    else:
                        action = np.array([0.0])
                actions.append(action)
            
            # Apply perturbation if active
            if perturbation_active:
                if perturbation_step < perturbation_duration:
                    perturbed_agent = np.random.randint(0, len(actions))
                    actions[perturbed_agent] = np.array([-4.0])
                    perturbation_step += 1
                else:
                    perturbation_active = False
            elif np.random.random() < perturbation_prob:
                perturbation_active = True
                perturbation_step = 0
                episode_data['perturbations_triggered'] += 1
            
            # Step environment
            observations, rewards, terminated, truncated, info = env.step(actions)
            
            # Record metrics
            for i, vehicle in enumerate(env.vehicles):
                episode_data['velocities'][i].append(vehicle.velocity)
            
            # Calculate distances
            distances = []
            for i in range(len(env.vehicles) - 1):
                pos1 = env.vehicles[i].position
                pos2 = env.vehicles[i + 1].position
                distance = env._calculate_distance(pos1, pos2)
                distances.append(distance)
            
            if distances:
                episode_data['distances'].append(np.mean(distances))
            
            # Check for collisions (after warmup)
            if step_count >= 50:
                if hasattr(env, '_check_safety_violations'):
                    if env._check_safety_violations():
                        episode_data['collisions'] += 1
            
            episode_data['total_reward'] += np.sum(rewards)
            episode_data['step_count'] = step_count
            
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Calculate metrics
        avg_velocity = np.mean([np.mean(v) for v in episode_data['velocities'] if v])
        distance_std = np.std(episode_data['distances']) if episode_data['distances'] else 0.0
        
        return {
            'avg_velocity': avg_velocity,
            'distance_std': distance_std,
            'collisions': episode_data['collisions'],
            'total_reward': episode_data['total_reward'],
            'perturbations': episode_data['perturbations_triggered'],
            'steps': episode_data['step_count']
        }
    
    def evaluate_model_set(self, model_prefix='final_model', num_episodes=5):
        """Evaluate a set of trained models"""
        results = []
        
        # Load models
        models = {}
        for agent_id in range(NUM_VEHICLES):
            try:
                model_path = f"models/agent_{agent_id}/{model_prefix}"
                model = self.load_model(model_path)
                if model is not None:
                    models[agent_id] = model
                    print(f"Loaded model for Agent {agent_id}")
            except Exception as e:
                print(f"Failed to load model for Agent {agent_id}: {e}")
        
        if not models:
            print("No models loaded!")
            return pd.DataFrame()
        
        print(f"Loaded {len(models)}/{NUM_VEHICLES} models")
        
        # Create environment
        env = MultiAgentACCEnvironment()
        
        # Run evaluation episodes
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            episode_result = self.run_evaluation_episode(env, models)
            episode_result['episode'] = episode + 1
            
            print(f"  Avg velocity: {episode_result['avg_velocity']:.2f} m/s")
            print(f"  Collisions: {episode_result['collisions']}")
            print(f"  Perturbations: {episode_result['perturbations']}")
            
            results.append(episode_result)
        
        env.close()
        
        return pd.DataFrame(results)
    
    def create_evaluation_table(self, results_df):
        """Create formatted table of results"""
        if results_df.empty:
            return "No results to display"
        
        table = "="*60 + "\n"
        table += "EVALUATION RESULTS\n"
        table += "="*60 + "\n\n"
        
        table += f"Average Velocity: {results_df['avg_velocity'].mean():.2f} ± {results_df['avg_velocity'].std():.2f} m/s\n"
        table += f"Distance Std Dev: {results_df['distance_std'].mean():.2f} ± {results_df['distance_std'].std():.2f} m\n"
        table += f"Total Collisions: {results_df['collisions'].sum()}\n"
        table += f"Total Reward: {results_df['total_reward'].mean():.1f} ± {results_df['total_reward'].std():.1f}\n"
        table += f"Avg Perturbations: {results_df['perturbations'].mean():.1f}\n"
        table += f"Episodes: {len(results_df)}\n"
        
        return table
    
    def plot_results(self, results_df, save_path="evaluation_results.png"):
        """Create visualization of results"""
        if results_df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # Plot velocity over episodes
        axes[0, 0].plot(results_df['episode'], results_df['avg_velocity'], 'o-')
        axes[0, 0].axhline(y=DESIRED_VELOCITY, color='r', linestyle='--', label='Target')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Velocity (m/s)')
        axes[0, 0].set_title('Average Velocity per Episode')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot distance variability
        axes[0, 1].plot(results_df['episode'], results_df['distance_std'], 'o-', color='orange')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Distance Std Dev (m)')
        axes[0, 1].set_title('Distance Variability')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot rewards
        axes[1, 0].plot(results_df['episode'], results_df['total_reward'], 'o-', color='green')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].set_title('Reward per Episode')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot collisions and perturbations
        axes[1, 1].bar(results_df['episode'], results_df['collisions'], alpha=0.6, label='Collisions')
        axes[1, 1].bar(results_df['episode'], results_df['perturbations'], alpha=0.6, label='Perturbations')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Collisions and Perturbations')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results plot saved to {save_path}")

def main():
    """Main evaluation function"""
    print("Starting Model Evaluation")
    print("="*60)
    
    evaluator = ModelEvaluator()
    
    # Run evaluation
    results_df = evaluator.evaluate_model_set(
        model_prefix='best_model',
        num_episodes=5
    )
    
    if results_df.empty:
        print("No evaluation results generated!")
        return
    
    # Display results
    table = evaluator.create_evaluation_table(results_df)
    print("\n" + table)
    
    # Save results
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nResults saved to evaluation_results.csv")
    
    # Create visualization
    evaluator.plot_results(results_df)

if __name__ == "__main__":
    main()
