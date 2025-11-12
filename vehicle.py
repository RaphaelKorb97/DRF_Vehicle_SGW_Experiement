"""Vehicle dynamics model for the car-following simulation"""
import numpy as np
from config import *

class Vehicle:
    """Represents a single vehicle with longitudinal dynamics"""
    
    def __init__(self, vehicle_id, initial_position=0.0, initial_velocity=0.0):
        self.id = vehicle_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.acceleration = 0.0
        self.is_rl_controlled = True
        
        # History for comfort calculations
        self.acceleration_history = []
        
    def update(self, acceleration, dt=SIMULATION_DT):
        """Update vehicle state based on acceleration command"""
        acceleration = np.clip(acceleration, MAX_DECELERATION, MAX_ACCELERATION)
        
        self.acceleration = acceleration
        self.velocity += acceleration * dt
        self.velocity = np.clip(self.velocity, MIN_VELOCITY, MAX_VELOCITY)
        self.position += self.velocity * dt
        
        # Handle periodic boundaries
        if self.position >= ROAD_LENGTH:
            self.position -= ROAD_LENGTH
        elif self.position < 0:
            self.position += ROAD_LENGTH
            
        self.acceleration_history.append(acceleration)
        if len(self.acceleration_history) > 10:
            self.acceleration_history.pop(0)
    
    def get_comfort_metric(self):
        """Calculate comfort metric based on jerk (rate of change of acceleration)"""
        if len(self.acceleration_history) < 2:
            return 0.0
        
        jerks = []
        for i in range(1, len(self.acceleration_history)):
            jerk = abs(self.acceleration_history[i] - self.acceleration_history[i-1]) / SIMULATION_DT
            jerks.append(jerk)
        
        return np.mean(jerks) if jerks else 0.0
    
    def __repr__(self):
        return f"Vehicle(id={self.id}, pos={self.position:.1f}, vel={self.velocity:.1f})"
