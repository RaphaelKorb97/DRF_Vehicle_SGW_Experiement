"""Multi-Agent environment for training RL-controlled vehicles"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from vehicle import Vehicle
from config import *

class MultiAgentACCEnvironment(gym.Env):
    """Multi-agent environment where all vehicles are RL-controlled"""
    
    metadata = {'render_modes': ['human'], 'render_fps': 10}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.num_agents = NUM_VEHICLES
        
        # Action and observation spaces
        single_action_space = spaces.Box(
            low=MAX_DECELERATION, 
            high=MAX_ACCELERATION, 
            shape=(1,), 
            dtype=np.float32
        )
        
        single_observation_space = spaces.Box(
            low=np.array([0.0, -OBSERVATION_RANGE, -MAX_VELOCITY*2, 0.0], dtype=np.float32),
            high=np.array([MAX_VELOCITY, OBSERVATION_RANGE, MAX_VELOCITY*2, MAX_VELOCITY], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = [single_action_space for _ in range(self.num_agents)]
        self.observation_space = [single_observation_space for _ in range(self.num_agents)]
        self.single_action_space = single_action_space
        self.single_observation_space = single_observation_space
        
        self.vehicles = []
        self.step_count = 0
        self.agent_rewards = np.zeros(self.num_agents)
        
        # Noise simulation for sensor inaccuracies
        self.noise_enabled = False
        self.noise_std = 0.5
        
    def reset(self, seed=None, options=None):
        """Reset the environment with all vehicles as RL agents"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.vehicles = []
        self.step_count = 0
        self.agent_rewards = np.zeros(self.num_agents)
        
        # Space vehicles evenly around the circular track
        base_spacing = ROAD_LENGTH / self.num_agents
        positions = []
        
        for i in range(self.num_agents):
            position = i * base_spacing
            position += np.random.uniform(-2.0, 2.0)  # Small random offset
            position = position % ROAD_LENGTH
            positions.append(position)
        
        positions = sorted(positions)
        
        for i in range(self.num_agents):
            initial_velocity = np.random.uniform(8.0, 12.0)
            vehicle = Vehicle(i, positions[i], initial_velocity)
            self.vehicles.append(vehicle)
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions):
        """Execute one time step with actions from all agents"""
        if isinstance(actions, np.ndarray):
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
        else:
            actions = np.array(actions).reshape(-1, 1)
        
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
        
        # Apply safety override to prevent collisions
        safe_actions = self._apply_safety_override(actions)
        
        # Update all vehicles
        for i, vehicle in enumerate(self.vehicles):
            acceleration = float(safe_actions[i])
            vehicle.update(acceleration)
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        self.agent_rewards += rewards
        
        # Check termination
        terminated = self._check_safety_violations()
        truncated = self.step_count >= 5000
        
        observations = self._get_observations()
        info = self._get_info()
        
        self.step_count += 1
        
        return observations, rewards, terminated, truncated, info
    
    def _get_observations(self):
        """Get observations for all agents"""
        observations = []
        
        for vehicle in self.vehicles:
            front_distance, front_velocity, relative_velocity = self._get_front_vehicle_info(vehicle)
            
            if front_distance is None:
                front_distance = OBSERVATION_RANGE
                front_velocity = DESIRED_VELOCITY
                relative_velocity = 0.0
            
            obs = np.array([
                vehicle.velocity,
                front_distance,
                relative_velocity,
                front_velocity
            ], dtype=np.float32)
            
            observations.append(obs)
        
        return observations
    
    def _get_front_vehicle_info(self, vehicle):
        """Get information about the vehicle in front"""
        min_distance = float('inf')
        front_vehicle = None
        
        for other_vehicle in self.vehicles:
            if other_vehicle.id == vehicle.id:
                continue
                
            distance = self._calculate_distance(vehicle.position, other_vehicle.position)
            
            if distance > 0 and distance < min_distance and distance <= OBSERVATION_RANGE:
                min_distance = distance
                front_vehicle = other_vehicle
        
        if front_vehicle is None:
            return None, None, None
        
        # Add Gaussian noise if enabled
        if self.noise_enabled:
            noise = np.random.normal(0, self.noise_std)
            min_distance = max(0.1, min_distance + noise)
        
        relative_velocity = vehicle.velocity - front_vehicle.velocity
        return min_distance, front_vehicle.velocity, relative_velocity
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate distance considering periodic boundaries"""
        if pos2 >= pos1:
            forward_distance = pos2 - pos1
        else:
            forward_distance = (ROAD_LENGTH - pos1) + pos2
        
        return forward_distance
    
    def _apply_safety_override(self, actions):
        """Apply safety override to prevent collisions"""
        safe_actions = actions.copy()
        
        for i, vehicle in enumerate(self.vehicles):
            front_distance, front_velocity, relative_velocity = self._get_front_vehicle_info(vehicle)
            
            if front_distance is not None:
                # Emergency braking for critical situations
                if front_distance < MIN_DISTANCE and relative_velocity > 0:
                    emergency_brake = MAX_DECELERATION * 0.7
                    safe_actions[i] = emergency_brake
                elif front_distance < MIN_DISTANCE * 1.5 and relative_velocity > 2.0:
                    if safe_actions[i] > 0.5:
                        safe_actions[i] = min(0.5, safe_actions[i])
        
        return safe_actions
    
    def _calculate_rewards(self):
        """Calculate individual rewards with safety-first approach"""
        rewards = []
        
        for vehicle in self.vehicles:
            reward = 0.0
            
            # Velocity reward (maintain desired velocity)
            velocity_error = abs(vehicle.velocity - DESIRED_VELOCITY)
            velocity_reward = -velocity_error / DESIRED_VELOCITY
            reward += REWARD_VELOCITY_WEIGHT * velocity_reward
            
            # Get front vehicle info for distance-based rewards
            front_distance, front_velocity, relative_velocity = self._get_front_vehicle_info(vehicle)
            
            # Distance reward (maintain desired headway d*(t) = T*v(t) + ℓ)
            if front_distance is not None:
                desired_headway = DESIRED_TIME_HEADWAY * vehicle.velocity + VEHICLE_LENGTH
                if desired_headway > 0:  # Avoid division by zero
                    distance_error = abs(front_distance - desired_headway) / desired_headway
                    distance_reward = -distance_error
                    reward += REWARD_DISTANCE_WEIGHT * distance_reward
            
            # Safety rewards (progressive zones)
            if front_distance is not None:
                safety_reward = self._calculate_progressive_safety_reward(front_distance)
                reward += safety_reward
            
            # Collision penalty
            if front_distance is not None and front_distance < MIN_DISTANCE * 0.5:
                emergency_penalty = REWARD_COLLISION_PENALTY * ((MIN_DISTANCE * 0.5 - front_distance) / (MIN_DISTANCE * 0.5)) ** 2
                reward += emergency_penalty
            
            rewards.append(reward)
        
        return np.array(rewards)
    
    def _calculate_progressive_safety_reward(self, front_distance):
        """Calculate safety reward based on distance zones"""
        if front_distance >= SAFETY_ZONE_EXCELLENT:
            return EXCELLENT_SPACING_BONUS
        elif front_distance >= SAFETY_ZONE_GOOD:
            return GOOD_SPACING_BONUS
        elif front_distance >= SAFETY_ZONE_ACCEPTABLE:
            return 0.0
        elif front_distance >= SAFETY_ZONE_CAUTION:
            penalty_ratio = (SAFETY_ZONE_ACCEPTABLE - front_distance) / (SAFETY_ZONE_ACCEPTABLE - SAFETY_ZONE_CAUTION)
            return CAUTION_PENALTY_MULTIPLIER * penalty_ratio
        elif front_distance >= SAFETY_ZONE_WARNING:
            penalty_ratio = (SAFETY_ZONE_CAUTION - front_distance) / (SAFETY_ZONE_CAUTION - SAFETY_ZONE_WARNING)
            return WARNING_PENALTY_MULTIPLIER * penalty_ratio
        else:
            penalty_ratio = (SAFETY_ZONE_WARNING - front_distance) / SAFETY_ZONE_WARNING
            return CRITICAL_PENALTY_MULTIPLIER * (penalty_ratio ** 2)
    
    def _check_safety_violations(self):
        """Check for serious safety violations"""
        for i, vehicle1 in enumerate(self.vehicles):
            for j, vehicle2 in enumerate(self.vehicles[i+1:], i+1):
                distance = min(
                    self._calculate_distance(vehicle1.position, vehicle2.position),
                    self._calculate_distance(vehicle2.position, vehicle1.position)
                )
                
                if distance < MIN_DISTANCE * 0.5:
                    return True
        
        return False
    
    def _get_info(self):
        """Get additional info about environment state"""
        info = {
            'step_count': self.step_count,
            'agent_rewards': self.agent_rewards.copy(),
            'vehicles': []
        }
        
        for vehicle in self.vehicles:
            front_distance, _, _ = self._get_front_vehicle_info(vehicle)
            vehicle_info = {
                'id': vehicle.id,
                'position': vehicle.position,
                'velocity': vehicle.velocity,
                'acceleration': vehicle.acceleration,
                'front_distance': front_distance,
                'comfort_metric': vehicle.get_comfort_metric()
            }
            info['vehicles'].append(vehicle_info)
        
        return info
    
    def render(self):
        """Render the environment (optional - implement if needed)"""
        pass
    
    def close(self):
        """Close the environment"""
        pass


class SingleAgentWrapper(gym.Env):
    """Wrapper to train individual agents from multi-agent environment"""
    
    def __init__(self, multi_env, agent_id):
        super().__init__()
        self.multi_env = multi_env
        self.agent_id = agent_id
        
        self.action_space = multi_env.single_action_space
        self.observation_space = multi_env.single_observation_space
        self.other_agent_actions = [np.array([0.0]) for _ in range(multi_env.num_agents)]
    
    @property
    def noise_enabled(self):
        return self.multi_env.noise_enabled
    
    @noise_enabled.setter
    def noise_enabled(self, value):
        self.multi_env.noise_enabled = value
    
    @property
    def noise_std(self):
        return self.multi_env.noise_std
    
    @noise_std.setter
    def noise_std(self, value):
        self.multi_env.noise_std = value
        
    def reset(self, **kwargs):
        obs_list, info = self.multi_env.reset(**kwargs)
        return obs_list[self.agent_id], info
    
    def step(self, action):
        # Create actions for all agents
        actions = []
        for i in range(self.multi_env.num_agents):
            if i == self.agent_id:
                actions.append(action)
            else:
                # Simple rule-based action for other agents
                other_vehicle = self.multi_env.vehicles[i]
                if other_vehicle.velocity < DESIRED_VELOCITY * 0.9:
                    actions.append(np.array([1.0]))
                elif other_vehicle.velocity > DESIRED_VELOCITY * 1.1:
                    actions.append(np.array([-0.5]))
                else:
                    actions.append(np.array([0.0]))
        
        obs_list, rewards, terminated, truncated, info = self.multi_env.step(actions)
        
        return obs_list[self.agent_id], rewards[self.agent_id], terminated, truncated, info
    
    def render(self, **kwargs):
        return self.multi_env.render(**kwargs)
    
    def close(self):
        return self.multi_env.close()
