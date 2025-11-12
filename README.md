# Deep Reinforcement Learning for Stop-and-Go Wave Mitigation

This repository contains the implementation of a multi-agent deep reinforcement learning system for mitigating stop-and-go waves in traffic flow, as described in our research paper.

## Overview

We train autonomous vehicles using deep reinforcement learning to stabilize traffic flow on a circular road. Each vehicle learns an individual policy to maintain safe distances while achieving desired velocities, leading to emergent cooperative strategies that suppress stop-and-go oscillations.

### Key Features

- **Multi-agent setup**: 15-25 vehicles with independent RL policies
- **Ring road topology**: Circular 220m track inspired by the Sugiyama experiment
- **PPO algorithm**: Proximal Policy Optimization with 3-layer neural network
- **Robust training**: Random perturbations and sensor noise during training
- **Safety-first approach**: Progressive reward zones and collision avoidance

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 2.1.0+
- Gymnasium 0.29.1
- Stable-Baselines3 2.2.1
- NumPy, Matplotlib

## Project Structure

```
├── config.py                    # Configuration parameters
├── vehicle.py                   # Vehicle dynamics model
├── multi_agent_environment.py   # Gymnasium environment
├── multi_agent_train.py         # Training script
├── evaluation.py                # Evaluation and testing
├── visual_demo.py              # Visualization (optional)
└── requirements.txt            # Dependencies
```

## Quick Start

### Training

Train all agents sequentially:

```bash
python multi_agent_train.py --mode train
```

Train a specific agent:

```bash
python multi_agent_train.py --mode train --agent_id 0
```

### Evaluation

Evaluate trained models:

```bash
python multi_agent_train.py --mode eval --episodes 10
```

Run comprehensive evaluation with metrics:

```bash
python evaluation.py
```

## Environment Setup

### State Space (4-dimensional)
- Own velocity \(v_t\)
- Relative distance to front vehicle \(d_t\)
- Relative velocity to front vehicle \(\Delta v_t\)
- Front vehicle velocity \(v_{t,f}\)

### Action Space
- Continuous acceleration command: \(a_t \in [-5, 2]\) m/s²

### Reward Function

The reward balances multiple objectives:

$$r(t) = w_v r_v(t) + w_s r_s(t)$$

Where:
- **Velocity reward** \(r_v(t)\): Penalizes deviation from desired velocity (10 m/s)
- **Safety reward** \(r_s(t)\): Progressive penalties/bonuses based on distance zones

### Training Features

- **Perturbations**: Random emergency braking (3% probability per step, 3s duration)
- **Sensor noise**: Gaussian noise (σ = 0.5m) added to distance measurements
- **Safety override**: Prevents collisions during training

## Configuration

Key parameters in `config.py`:

```python
ROAD_LENGTH = 220.0          # Ring road length (meters)
NUM_VEHICLES = 15            # Number of vehicles
DESIRED_VELOCITY = 10.0      # Target velocity (m/s)
DESIRED_TIME_HEADWAY = 1.2   # Time gap parameter (s)
MIN_DISTANCE = 8.0           # Minimum safe distance (m)
```

## Results

Our trained agents exhibit emergent cooperative behavior where one vehicle maintains a large gap (buffer vehicle) while others maintain smaller time gaps, effectively stabilizing traffic flow and suppressing stop-and-go waves.

### Typical Performance
- **Average velocity**: ~10 m/s (target achieved)
- **Collision rate**: Near-zero with safety override
- **String stability**: Improved wave damping compared to traditional models

## Pre-trained Models

Pre-trained model weights are not included in this repository. If you are interested in obtaining the trained models for evaluation or further research, please contact the authors via email (see Contact section below).

## GPU Acceleration

The code automatically detects and uses available GPU acceleration:
- **Apple Silicon**: Metal Performance Shaders (MPS)
- **NVIDIA**: CUDA
- **Fallback**: CPU

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Deep Reinforcement Learning for Stop-and-Go Wave Mitigation},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Environment implementation based on Gymnasium framework
- PPO implementation from Stable-Baselines3
- Inspired by the Sugiyama experiment on traffic oscillations

## Contact

For questions, issues, or to request pre-trained models, please open an issue on GitHub or contact the authors directly.

