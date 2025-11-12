"""Simple Visual Demo - visualize trained agents on circular track"""
import pygame
import math
import numpy as np
import os
from stable_baselines3 import PPO
from multi_agent_environment import MultiAgentACCEnvironment
from config import *

# Initialize pygame
pygame.init()

# Display settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 59, 48)
BLUE = (0, 122, 255)
GREEN = (52, 199, 89)

# Track settings
TRACK_CENTER_X = WINDOW_WIDTH // 2 - 150
TRACK_CENTER_Y = WINDOW_HEIGHT // 2
TRACK_RADIUS = 250

# Vehicle colors
VEHICLE_COLORS = [
    (255, 59, 48), (0, 122, 255), (52, 199, 89), (255, 149, 0),
    (175, 82, 222), (255, 45, 85), (90, 200, 250), (88, 86, 214),
    (0, 199, 190), (255, 204, 0), (255, 102, 102), (102, 178, 255),
    (102, 255, 178), (255, 178, 102), (178, 102, 255)
]

class VisualDemo:
    """Simple visualization of multi-agent car-following"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Multi-Agent Car-Following Simulation")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.title_font = pygame.font.SysFont("helvetica", 24, bold=True)
        self.body_font = pygame.font.SysFont("helvetica", 16)
        self.small_font = pygame.font.SysFont("helvetica", 14)
        
        # Initialize environment
        self.env = MultiAgentACCEnvironment()
        self.models = self.load_models()
        
        # Simulation state
        self.running = True
        self.paused = False
        self.step_count = 0
        self.observations = None
        
        # Metrics
        self.avg_velocity = 0.0
        self.min_distance = 0.0
        self.collision_count = 0
        
        # Enhanced observation history for reward5 models
        self.observation_history = {i: [] for i in range(NUM_VEHICLES)}
        self.obs_history_length = 3
        
        # Reset simulation
        self.reset_simulation()
    
    def load_models(self):
        """Load trained models for all agents"""
        models = {}
        for agent_id in range(NUM_VEHICLES):
            try:
                # Try different model paths
                for suffix in ["reward5", "final_model"]:
                    model_path = f"models/agent_{agent_id}_{suffix}"
                    if os.path.exists(model_path + ".zip"):
                        models[agent_id] = PPO.load(model_path)
                        print(f"Loaded model for Agent {agent_id}: {suffix}")
                        break
            except Exception as e:
                print(f"Could not load model for Agent {agent_id}: {e}")
                models[agent_id] = None
        
        print(f"Loaded {len([m for m in models.values() if m is not None])}/{NUM_VEHICLES} models")
        return models
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.observations, _ = self.env.reset()
        self.step_count = 0
        self.collision_count = 0
        # Clear observation history
        self.observation_history = {i: [] for i in range(NUM_VEHICLES)}
        print("Simulation reset")
    
    def create_enhanced_observation(self, agent_id, base_obs):
        """Create enhanced 16D observation from 4D base observation (for reward5 models)"""
        # Store observation history
        self.observation_history[agent_id].append(base_obs.copy())
        
        # Keep only recent history
        if len(self.observation_history[agent_id]) > self.obs_history_length:
            self.observation_history[agent_id].pop(0)
        
        # Pad history if needed (for early steps)
        while len(self.observation_history[agent_id]) < self.obs_history_length:
            self.observation_history[agent_id].insert(0, base_obs.copy())
        
        # Flatten observation history (3 timesteps × 4 features = 12 features)
        obs_history_flat = np.concatenate(self.observation_history[agent_id])
        
        # Extract features from current observation
        ego_velocity = base_obs[0]
        front_distance = base_obs[1]
        relative_velocity = base_obs[2]
        front_velocity = base_obs[3]
        
        # Calculate additional features (matching training)
        time_to_collision = front_distance / max(relative_velocity, 0.1) if relative_velocity > 0 else 100.0
        desired_distance = ego_velocity * DESIRED_TIME_HEADWAY + MIN_DISTANCE
        distance_error = front_distance - desired_distance
        velocity_efficiency = 1.0 - abs(ego_velocity - DESIRED_VELOCITY) / DESIRED_VELOCITY
        
        additional_features = np.array([
            time_to_collision / 10.0,  # Normalized TTC
            distance_error / 50.0,     # Normalized distance error
            velocity_efficiency,        # Velocity efficiency (0-1)
            0.0                        # Curriculum difficulty (0 for evaluation)
        ], dtype=np.float32)
        
        # Combine: 12 (history) + 4 (additional) = 16 dimensions
        enhanced_obs = np.concatenate([obs_history_flat, additional_features])
        
        return enhanced_obs
    
    def step_simulation(self):
        """Execute one simulation step"""
        if self.paused:
            return
        
        # Get actions from models
        actions = []
        for agent_id in range(NUM_VEHICLES):
            if agent_id in self.models and self.models[agent_id] is not None:
                try:
                    base_obs = self.observations[agent_id]
                    # Create enhanced observation (16D) for reward5 models
                    enhanced_obs = self.create_enhanced_observation(agent_id, base_obs)
                    action, _ = self.models[agent_id].predict(
                        enhanced_obs, 
                        deterministic=True
                    )
                except Exception as e:
                    print(f"Error getting action for agent {agent_id}: {e}")
                    action = np.array([0.0])
            else:
                # Fallback: simple rule-based action
                obs = self.observations[agent_id]
                if obs[1] < 15.0:  # Front distance < 15m
                    action = np.array([-2.0])
                elif obs[0] < DESIRED_VELOCITY * 0.9:
                    action = np.array([1.0])
                else:
                    action = np.array([0.0])
            actions.append(action)
        
        # Step environment
        self.observations, rewards, terminated, truncated, info = self.env.step(actions)
        self.step_count += 1
        
        # Update metrics
        velocities = [v.velocity for v in self.env.vehicles]
        self.avg_velocity = np.mean(velocities)
        
        # Calculate minimum distance
        min_dist = float('inf')
        for i in range(len(self.env.vehicles)):
            front_dist, _, _ = self.env._get_front_vehicle_info(self.env.vehicles[i])
            if front_dist is not None:
                min_dist = min(min_dist, front_dist)
        self.min_distance = min_dist if min_dist != float('inf') else 0.0
        
        # Check collisions
        if self.env._check_safety_violations():
            self.collision_count += 1
        
        # Reset if terminated
        if terminated or truncated:
            print("Episode ended, resetting...")
            self.reset_simulation()
    
    def pos_to_screen(self, position):
        """Convert road position to screen coordinates"""
        angle = (position / ROAD_LENGTH) * 2 * math.pi - math.pi / 2
        x = TRACK_CENTER_X + TRACK_RADIUS * math.cos(angle)
        y = TRACK_CENTER_Y + TRACK_RADIUS * math.sin(angle)
        return int(x), int(y)
    
    def draw_track(self):
        """Draw the circular track"""
        # Outer circle
        pygame.draw.circle(self.screen, GRAY, 
                          (TRACK_CENTER_X, TRACK_CENTER_Y), 
                          TRACK_RADIUS + 30, 2)
        # Inner circle
        pygame.draw.circle(self.screen, GRAY, 
                          (TRACK_CENTER_X, TRACK_CENTER_Y), 
                          TRACK_RADIUS - 30, 2)
        # Center line
        pygame.draw.circle(self.screen, GRAY, 
                          (TRACK_CENTER_X, TRACK_CENTER_Y), 
                          TRACK_RADIUS, 1)
    
    def draw_vehicles(self):
        """Draw all vehicles"""
        for i, vehicle in enumerate(self.env.vehicles):
            x, y = self.pos_to_screen(vehicle.position)
            color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
            
            # Draw vehicle as circle
            pygame.draw.circle(self.screen, color, (x, y), 12)
            pygame.draw.circle(self.screen, BLACK, (x, y), 12, 2)
            
            # Draw velocity indicator
            angle = (vehicle.position / ROAD_LENGTH) * 2 * math.pi - math.pi / 2
            speed_scale = vehicle.velocity / MAX_VELOCITY
            indicator_length = 20 + int(speed_scale * 20)
            end_x = x + indicator_length * math.cos(angle + math.pi / 2)
            end_y = y + indicator_length * math.sin(angle + math.pi / 2)
            pygame.draw.line(self.screen, color, (x, y), (end_x, end_y), 3)
    
    def draw_metrics(self):
        """Draw metrics panel"""
        panel_x = WINDOW_WIDTH - 320
        panel_y = 20
        panel_width = 300
        panel_height = 250
        
        # Draw panel background
        pygame.draw.rect(self.screen, WHITE, 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, GRAY, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.title_font.render("Simulation Metrics", True, BLACK)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Metrics
        y_offset = panel_y + 50
        line_height = 30
        
        metrics = [
            f"Step: {self.step_count}",
            f"Time: {self.step_count * SIMULATION_DT:.1f}s",
            f"Avg Velocity: {self.avg_velocity:.2f} m/s",
            f"Target: {DESIRED_VELOCITY:.2f} m/s",
            f"Min Distance: {self.min_distance:.2f} m",
            f"Collisions: {self.collision_count}",
        ]
        
        for i, metric in enumerate(metrics):
            text = self.body_font.render(metric, True, BLACK)
            self.screen.blit(text, (panel_x + 15, y_offset + i * line_height))
    
    def draw_controls(self):
        """Draw control instructions"""
        panel_x = WINDOW_WIDTH - 320
        panel_y = 290
        panel_width = 300
        panel_height = 150
        
        # Draw panel background
        pygame.draw.rect(self.screen, WHITE, 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, GRAY, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.title_font.render("Controls", True, BLACK)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Controls
        y_offset = panel_y + 50
        line_height = 25
        
        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Simulation",
            "Q/ESC: Quit",
        ]
        
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, BLACK)
            self.screen.blit(text, (panel_x + 15, y_offset + i * line_height))
    
    def draw_title(self):
        """Draw main title"""
        title = self.title_font.render("Multi-Agent Car-Following", True, BLACK)
        subtitle = self.body_font.render(
            f"{NUM_VEHICLES} RL-controlled vehicles | {ROAD_LENGTH:.0f}m circular track", 
            True, GRAY
        )
        
        self.screen.blit(title, (20, 20))
        self.screen.blit(subtitle, (20, 50))
        
        # Draw pause indicator
        if self.paused:
            pause_text = self.title_font.render("PAUSED", True, RED)
            self.screen.blit(pause_text, (20, 80))
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")
                
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
    
    def render(self):
        """Render the simulation"""
        self.screen.fill(WHITE)
        
        self.draw_title()
        self.draw_track()
        self.draw_vehicles()
        self.draw_metrics()
        self.draw_controls()
        
        pygame.display.flip()
    
    def run(self):
        """Main simulation loop"""
        print("\n" + "="*60)
        print("Multi-Agent Car-Following Visual Demo")
        print("="*60)
        print(f"Vehicles: {NUM_VEHICLES}")
        print(f"Models loaded: {len([m for m in self.models.values() if m is not None])}")
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset")
        print("  Q/ESC: Quit")
        print("="*60 + "\n")
        
        while self.running:
            self.handle_events()
            self.step_simulation()
            self.render()
            self.clock.tick(FPS)
        
        # Cleanup
        self.env.close()
        pygame.quit()
        
        print("\nSimulation ended")
        print(f"Final stats:")
        print(f"  Steps: {self.step_count}")
        print(f"  Time: {self.step_count * SIMULATION_DT:.1f}s")
        print(f"  Avg velocity: {self.avg_velocity:.2f} m/s")
        print(f"  Collisions: {self.collision_count}")

if __name__ == "__main__":
    demo = VisualDemo()
    demo.run()
