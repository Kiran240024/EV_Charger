"""
Train a Deep Q-Network for EV charging optimization.

Uses the existing EVChargingEnv and DQNAgent.
"""

from pathlib import Path

import numpy as np

from agents.dqn import DQNAgent
from environment.ev_charging_env import EVChargingEnv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EPISODES = 5
EPISODE_LENGTH = 96       # 24 hours, with 15-minute intervals
SEED = 42

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "dqn_model.pth"


def main():
    # Check that the processed dataset exists.
    csv_path = Path("data/processed/ev_charging_processed.csv")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path.resolve()}"
        )

    # Create the environment.
    env = EVChargingEnv(
        csv_path=str(csv_path),
        episode_len=EPISODE_LENGTH,
        seed=SEED,
    )

    # Create the DQN agent.
    agent = DQNAgent(
        state_size=env.STATE_SIZE,
        n_actions=env.N_ACTIONS,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=10000,
        target_update=100,
        seed=SEED,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting DQN training")
    print(f"Dataset rows: {env.T}")
    print(f"State size: {env.STATE_SIZE}")
    print(f"Number of actions: {env.N_ACTIONS}")
    print(f"Episodes: {EPISODES}")
    print(f"Steps per episode: {EPISODE_LENGTH}")
    print("-" * 50)

    # Training loop.
    for episode in range(1, EPISODES + 1):
        state = env.reset()

        total_reward = 0.0
        total_cost = 0.0
        total_energy = 0.0
        losses = []

        done = False

        while not done:
            # Choose an action.
            action = agent.select_action(state, explore=True)

            # Execute the action in the environment.
            next_state, reward, done, info = env.step(action)

            # Store the transition.
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # Learn from previous experiences.
            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

            # Accumulate metrics.
            total_reward += reward
            total_cost += info["charging_cost"]
            total_energy += info["energy_kwh"]

            state = next_state

        # Reduce exploration after each episode.
        agent.decay_epsilon()

        average_loss = (
            float(np.mean(losses)) if losses else 0.0
        )

        print(
            f"Episode {episode}/{EPISODES} | "
            f"Reward: {total_reward:.3f} | "
            f"Cost: {total_cost:.3f} | "
            f"Energy: {total_energy:.2f} kWh | "
            f"Avg loss: {average_loss:.5f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    # Save the trained model.
    agent.save(str(MODEL_PATH))

    print("-" * 50)
    print(f"Training finished. Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()