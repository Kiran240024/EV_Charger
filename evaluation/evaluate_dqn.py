"""
Evaluate a trained DQN agent on the EV charging environment.
"""

from pathlib import Path

import numpy as np

from agents.dqn import DQNAgent
from environment.ev_charging_env import EVChargingEnv


def main():
    dataset_path = Path(
        "data/processed/ev_charging_processed.csv"
    )

    model_path = Path("models/dqn_model.pth")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    # Use the same environment configuration as training.
    env = EVChargingEnv(
        csv_path=str(dataset_path),
        episode_len=96,
        seed=100,
    )

    agent = DQNAgent(
        state_size=env.STATE_SIZE,
        n_actions=env.N_ACTIONS,
        seed=42,
    )

    agent.load(str(model_path))

    # Disable exploration for evaluation.
    agent.epsilon = 0.0

    state = env.reset()

    total_reward = 0.0
    total_cost = 0.0
    total_energy = 0.0

    peak_grid_usage = float("-inf")
    total_overload = 0.0

    action_counts = np.zeros(
        env.N_ACTIONS,
        dtype=int,
    )

    done = False

    while not done:
        action = agent.select_action(
            state,
            explore=False,
        )

        action_counts[action] += 1

        next_state, reward, done, info = env.step(
            action
        )

        total_reward += reward
        total_cost += info["charging_cost"]
        total_energy += info["energy_kwh"]

        peak_grid_usage = max(
            peak_grid_usage,
            info["grid_usage_kw"],
        )

        total_overload += info["overload_kw"]

        state = next_state

    print("\nDQN Evaluation Results")
    print("=" * 40)

    print(f"Total reward: {total_reward:.3f}")
    print(f"Total charging cost: {total_cost:.3f}")
    print(f"Total energy charged: {total_energy:.2f} kWh")
    print(f"Peak grid usage: {peak_grid_usage:.2f} kW")
    print(f"Accumulated overload: {total_overload:.2f} kW")
    print(f"Average reward per step: {total_reward / 96:.3f}")

    print("\nAction distribution")
    print("-" * 40)

    for action, count in enumerate(action_counts):
        power_kw = action * env.ACTION_STEP_KW

        print(
            f"{power_kw:>3.0f} kW: "
            f"{count:>3} intervals"
        )


if __name__ == "__main__":
    main()