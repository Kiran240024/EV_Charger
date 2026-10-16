
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from environment.ev_charging_env import EVChargingEnv
from agents.q_learning import QLearningAgent
from evaluation.baselines import (
    never_charge,
    maximum_charge,
    cheap_only,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ev_charging_processed.csv"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "q_learning_model.npz"


def get_info_value(info, *keys, default=0.0):
    """Read a metric using the first matching info key."""
    for key in keys:
        if key in info:
            return float(info[key])
    return float(default)


def load_agent(env, model_path):
    """Load the saved Q-table and its discretization bins."""
    with np.load(model_path, allow_pickle=False) as model:
        q_table = model["Q"].copy()
        bins_json = model["bins"].item()

    bins = json.loads(str(bins_json))

    agent = QLearningAgent(env, epsilon=0.0, seed=42)
    agent.q_table = q_table
    agent.bins = bins
    agent.epsilon = 0.0

    if agent.q_table.shape[-1] != 12:
        raise ValueError("The loaded Q-table does not have 12 actions.")

    return agent


def evaluate_policy(
    policy_name,
    policy,
    data_path,
    start_indices,
    episode_len,
    price_threshold,
    model_path=None,
):
    """Evaluate a policy on identical, fixed data windows."""
    episode_results = []

    for start_idx in start_indices:
        env = EVChargingEnv(
            csv_path=str(data_path),
            episode_len=episode_len,
            seed=42,
        )

        agent = load_agent(env, model_path) if policy_name == "Q-learning" else None

        state = env.reset(start=start_idx)
        done = False

        metrics = {
            "reward": 0.0,
            "cost": 0.0,
            "grid_penalty": 0.0,
            "soc_penalty": 0.0,
            "energy": 0.0,
            "overload_intervals": 0,
            "overload_kw_sum": 0.0,
            "steps": 0,
        }

        while not done:
            if policy_name == "Q-learning":
                action = agent.select_action(state, explore=False)
            elif policy_name == "Cheap-only":
                action = policy(state, price_threshold)
            else:
                action = policy(state)

            next_state, reward, done, info = env.step(action)

            grid_usage = get_info_value(
                info, "grid_usage", "grid_usage_kw"
            )
            actual_energy = get_info_value(
                info,
                "energy_delivered",
                "energy_delivered_kwh",
                "energy_kwh",
            )

            metrics["reward"] += float(reward)
            metrics["cost"] += get_info_value(
                info, "charging_cost", "cost"
            )
            metrics["grid_penalty"] += get_info_value(
                info, "grid_penalty"
            )
            metrics["soc_penalty"] += get_info_value(
                info, "soc_penalty"
            )
            metrics["energy"] += actual_energy
            metrics["overload_intervals"] += int(
                grid_usage > env.GRID_LIMIT
            )
            metrics["overload_kw_sum"] += max(
                0.0, grid_usage - env.GRID_LIMIT
            )
            metrics["steps"] += 1

            state = next_state

        episode_results.append(metrics)

    return episode_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained EV charging Q-learning agent."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-len", type=int, default=672)
    args = parser.parse_args()

    if args.episodes < 1 or args.episode_len < 1:
        parser.error("--episodes and --episode-len must be positive.")

    if not args.data.exists():
        parser.error(f"Dataset not found: {args.data}")

    if not args.model.exists():
        parser.error(f"Trained model not found: {args.model}")

    df = pd.read_csv(args.data)
    price_columns = [
        column for column in df.columns if "price" in column.lower()
    ]

    if not price_columns:
        parser.error(
            "Could not find a price column in the dataset. "
            "Check the CSV column names before evaluating."
        )

    price_threshold = float(df[price_columns[0]].median())

    if len(df) < args.episode_len:
        parser.error("Episode length is longer than the dataset.")

    max_start = len(df) - args.episode_len
    start_indices = [
        min(i * args.episode_len, max_start)
        for i in range(args.episodes)
    ]

    policies = [
        ("Q-learning", None),
        ("Never charge", never_charge),
        ("Maximum charge", maximum_charge),
        ("Cheap-only", cheap_only),
    ]

    print("\nEV Charging Policy Evaluation")
    print("=" * 78)
    print(f"Dataset:          {args.data}")
    print(f"Model:            {args.model}")
    print(f"Evaluation windows: {args.episodes}")
    print(f"Steps per window:  {args.episode_len}")
    print(f"Cheap-only threshold (median price): {price_threshold:.6f}")
    print("=" * 78)

    results = {}

    for name, policy in policies:
        episodes = evaluate_policy(
            policy_name=name,
            policy=policy,
            data_path=args.data,
            start_indices=start_indices,
            episode_len=args.episode_len,
            price_threshold=price_threshold,
            model_path=args.model if name == "Q-learning" else None,
        )

        results[name] = {
            key: float(np.mean([episode[key] for episode in episodes]))
            for key in episodes[0]
        }

    print("\nMean results per evaluation window")
    print("-" * 108)
    print(
        f"{'Policy':<18}"
        f"{'Reward':>13}"
        f"{'Cost':>13}"
        f"{'Grid penalty':>15}"
        f"{'SOC penalty':>14}"
        f"{'Energy (kWh)':>16}"
        f"{'Overload steps':>17}"
    )
    print("-" * 108)

    for name, metrics in results.items():
        print(
            f"{name:<18}"
            f"{metrics['reward']:>13.2f}"
            f"{metrics['cost']:>13.2f}"
            f"{metrics['grid_penalty']:>15.2f}"
            f"{metrics['soc_penalty']:>14.2f}"
            f"{metrics['energy']:>16.2f}"
            f"{metrics['overload_intervals']:>17.2f}"
        )

    print("-" * 108)
    print("Higher reward is better. Compare cost and penalties alongside reward.")
    print("Results are averages across the same fixed evaluation windows.")


if __name__ == "__main__":
    main()
