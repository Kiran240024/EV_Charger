
"""
Train the tabular Q-learning agent for EV charging optimization.

Run from the project root:
    python3 -m training.train_q_learning

Quick test:
    python3 -m training.train_q_learning --episodes 10 --episode-len 96
"""

import argparse
import json
from pathlib import Path

import numpy as np

from environment.ev_charging_env import EVChargingEnv
from agents.q_learning import QLearningAgent


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ev_charging_processed.csv"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "q_learning_model.npz"
)


# --------------------------------------------------
# Training
# --------------------------------------------------

def train(
    csv_path,
    model_path,
    episodes=2000,
    episode_len=672,
    alpha=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    decay_fraction=0.8,
    seed=42,
):
    """Train the Q-learning agent and save the learned model."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    if episode_len < 1:
        raise ValueError("episode_len must be at least 1.")

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")

    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError(
            "Require 0 <= epsilon_end <= epsilon_start <= 1."
        )

    if not 0 < decay_fraction <= 1:
        raise ValueError("decay_fraction must be in (0, 1].")

    csv_path = Path(csv_path)
    model_path = Path(model_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {csv_path}"
        )

    # Create the simulation environment.
    env = EVChargingEnv(
        csv_path=str(csv_path),
        episode_len=episode_len,
        seed=seed,
    )

    # Initialize the tabular Q-learning agent.
    agent = QLearningAgent(
        env=env,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    episode_rewards = []
    episode_steps = []

    decay_episodes = max(1, int(episodes * decay_fraction))

    print("\nEV Charging Q-learning training")
    print("-" * 40)
    print(f"Dataset:       {csv_path}")
    print(f"Episodes:      {episodes}")
    print(f"Steps/episode: up to {episode_len}")
    print(f"Q-table shape: {agent.q_table.shape}")
    print(f"Learning rate: {alpha}")
    print(f"Discount:      {gamma}")
    print(f"Epsilon:       {epsilon_start} -> {epsilon_end}")
    print("-" * 40)

    report_every = max(1, episodes // 10)

    for episode in range(episodes):

        # Exponential epsilon decay, matching the old design.
        progress = min(episode / decay_episodes, 1.0)

        epsilon = epsilon_start * (
            epsilon_end / epsilon_start
        ) ** progress if epsilon_start > 0 else 0.0

        epsilon = max(epsilon_end, epsilon)
        agent.epsilon = epsilon

        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Select an action using epsilon-greedy.
            action = agent.select_action(state, explore=True)

            # Apply action to the simulation.
            next_state, reward, done, info = env.step(action)

            # Learn from the transition.
            agent.learn(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            state = next_state
            total_reward += float(reward)
            steps += 1

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if (
            (episode + 1) % report_every == 0
            or episode == 0
            or episode + 1 == episodes
        ):
            recent_rewards = episode_rewards[-min(100, len(episode_rewards)):]

            print(
                f"Episode {episode + 1:5d}/{episodes} | "
                f"epsilon={epsilon:.3f} | "
                f"reward={total_reward:,.2f} | "
                f"mean_recent_reward="
                f"{np.mean(recent_rewards):,.2f} | "
                f"steps={steps}"
            )

    # --------------------------------------------------
    # Save the Q-table and the information needed to use it
    # --------------------------------------------------

    model_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "bins": agent.bins,
        "hyperparameters": {
            "episodes": episodes,
            "episode_len": episode_len,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "decay_fraction": decay_fraction,
            "seed": seed,
        },
        "state_dimensions": [
            "connected_ev_count",
            "lowest_connected_soc",
            "electricity_price_level",
            "time_of_day_block",
            "grid_headroom_level",
        ],
        "n_actions": 12,
        "action_step_kw": 10.0,
    }

    np.savez_compressed(
        model_path,
        Q=agent.q_table,
        bins=json.dumps(agent.bins),
        metadata=json.dumps(metadata),
        episode_rewards=np.asarray(episode_rewards, dtype=np.float64),
        episode_steps=np.asarray(episode_steps, dtype=np.int32),
    )

    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Q-table shape: {agent.q_table.shape}")
    print(f"Final episode reward: {episode_rewards[-1]:,.2f}")
    print(
        "Last 100 episodes' mean reward: "
        f"{np.mean(episode_rewards[-100:]):,.2f}"
    )

    return agent, episode_rewards


# --------------------------------------------------
# Command-line entry point
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the EV charging Q-learning agent."
    )

    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--episode-len", type=int, default=672)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--decay-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        csv_path=args.data,
        model_path=args.output,
        episodes=args.episodes,
        episode_len=args.episode_len,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        decay_fraction=args.decay_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
