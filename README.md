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

### Visualization

Evaluate and visualize trained models:

```bash
python multi_agent_train.py --mode eval --episodes 10
```



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




## Citation

If you use this code in your research, please cite our paper:



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Environment implementation based on Gymnasium framework
- PPO implementation from Stable-Baselines3
- Inspired by the Sugiyama experiment on traffic oscillations

## Contact

For questions, issues, or to request pre-trained models, please open an issue on GitHub or contact the authors directly.

