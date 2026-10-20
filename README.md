# EV Charger: Reinforcement Learning for Smart EV Charging

## 1. Project Overview

This project explores reinforcement learning for intelligent electric vehicle
(EV) charging. It simulates multiple EVs connected to a charging system and
evaluates different charging strategies using charging cost, energy delivered,
grid loading, and battery state of charge (SOC).

The project includes a custom EV charging simulation environment, a
Q-learning agent, a DQN implementation, baseline charging strategies,
evaluation scripts, and automated tests.

## 2. Objectives

- Develop a simulation environment for multiple electric vehicles.
- Apply reinforcement learning to EV charging decisions.
- Evaluate charging strategies using consistent data windows.
- Measure charging cost, energy delivery, grid overload, and departure SOC.
- Compare learned policies against simple baseline strategies.
- Validate the environment and Q-learning implementation through automated tests.

## 3. Project Structure

```text
EV_Charger/
├── agents/
│   ├── q_learning.py
│   └── dqn.py
├── data/
│   ├── raw/
│   └── processed/
├── environment/
│   └── ev_charging_env.py
├── evaluation/
│   ├── baselines.py
│   ├── evaluate.py
│   ├── evaluate_dqn.py
│   ├── evaluation_results.csv
│   └── evaluation_results_200.csv
├── testing/
│   ├── test_environment.py
│   └── test_q_learning.py
├── training/
│   ├── train_q_learning.py
│   └── train_dqn.py
├── preprocess.py
├── requirements.txt
└── README.md# EV_Charger