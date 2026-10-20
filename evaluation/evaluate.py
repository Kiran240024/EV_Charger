"""
Evaluate EV charging policies on identical, reproducible data windows.

Policies:
    1. Q-learning
    2. Never charge
    3. Maximum charge
    4. Cheap-only

Run from the project root:
    python3 -m evaluation.evaluate

Example:
    python3 -m evaluation.evaluate \
        --model models/q_learning_model_200.npz \
        --episodes 5 \
        --episode-len 96 \
        --output evaluation/evaluation_results_200.csv
"""

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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ev_charging_processed.csv"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "q_learning_model.npz"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_results.csv"
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def get_info_value(info, *keys, default=0.0):
    """Return the first available metric as a float."""

    for key in keys:
        if key in info:
            value = float(info[key])

            if not np.isfinite(value):
                raise ValueError(
                    f"Environment metric '{key}' is not finite."
                )

            return value

    return float(default)


def load_agent(env, model_path):
    """Load a saved Q-learning model without exploration."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Q-learning model not found: {model_path}"
        )

    with np.load(model_path, allow_pickle=False) as model:
        required_arrays = {"Q", "bins"}

        missing = required_arrays.difference(model.files)

        if missing:
            raise ValueError(
                f"Model is missing required arrays: {sorted(missing)}"
            )

        q_table = model["Q"].copy()
        bins_json = model["bins"].item()

    try:
        bins = json.loads(str(bins_json))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            "The model contains invalid discretization-bin metadata."
        ) from exc

    agent = QLearningAgent(
        env,
        epsilon=0.0,
        seed=42,
    )

    if q_table.shape != agent.q_table.shape:
        raise ValueError(
            "The saved Q-table shape does not match the current agent. "
            f"Saved: {q_table.shape}; expected: {agent.q_table.shape}"
        )

    if not np.isfinite(q_table).all():
        raise ValueError("The saved Q-table contains invalid values.")

    agent.q_table = q_table
    agent.bins = bins
    agent.epsilon = 0.0

    return agent


def make_evaluation_windows(
    dataset_length,
    episode_len,
    number_of_windows,
):
    """
    Generate reproducible, non-overlapping evaluation windows.

    Windows are distributed across the available dataset. When enough
    data exists, each window begins where the previous window ends.
    """

    if episode_len < 1:
        raise ValueError("episode_len must be positive.")

    if number_of_windows < 1:
        raise ValueError("number_of_windows must be positive.")

    max_start = dataset_length - 1 - episode_len

    if max_start < 0:
        raise ValueError(
            "The dataset is too short for the requested episode length."
        )

    starts = [
        index * episode_len
        for index in range(number_of_windows)
    ]

    if starts[-1] > max_start:
        # Spread windows across the available range if the requested
        # count would otherwise exceed the dataset.
        starts = np.linspace(
            0,
            max_start,
            number_of_windows,
            dtype=int,
        ).tolist()

    # Prevent accidental duplicate windows when the dataset cannot
    # support the requested number of distinct starts.
    if len(set(starts)) != len(starts):
        raise ValueError(
            "The dataset cannot support this many distinct evaluation "
            "windows at the requested episode length. Reduce --episodes "
            "or --episode-len."
        )

    return starts


# ==========================================================
# POLICY EVALUATION
# ==========================================================

def evaluate_policy(
    policy_name,
    policy,
    data_path,
    start_indices,
    episode_len,
    price_threshold,
    model_path=None,
):
    """
    Evaluate a policy using identical, fixed dataset windows.

    Returns one dictionary of metrics per evaluation window.
    """

    episode_results = []

    for start_idx in start_indices:

        env = EVChargingEnv(
            csv_path=str(data_path),
            episode_len=episode_len,
            seed=42,
        )

        agent = None

        if policy_name == "Q-learning":
            if model_path is None:
                raise ValueError(
                    "A model path is required for Q-learning evaluation."
                )

            agent = load_agent(env, model_path)

        state = env.reset(start=int(start_idx))
        done = False

        metrics = {
            "reward": 0.0,
            "cost": 0.0,
            "energy_kwh": 0.0,
            "grid_penalty": 0.0,
            "soc_penalty": 0.0,
            "departure_penalty": 0.0,
            "peak_grid_kw": float("-inf"),
            "overload_intervals": 0,
            "overload_kw_sum": 0.0,
            "steps": 0,
            "departures": 0,
            "departures_below_90_soc": 0,
            "departure_soc_sum": 0.0,
        }

        while not done:

            if policy_name == "Q-learning":
                action = agent.select_action(
                    state,
                    explore=False,
                )

            elif policy_name == "Cheap-only":
                action = policy(
                    state,
                    price_threshold,
                )

            else:
                action = policy(state)

            next_state, reward, done, info = env.step(action)

            grid_usage = get_info_value(
                info,
                "grid_usage_kw",
                "grid_usage",
            )

            energy_kwh = get_info_value(
                info,
                "energy_kwh",
                "energy_delivered_kwh",
                "energy_delivered",
            )

            overload_kw = max(
                0.0,
                grid_usage - env.grid_limit,
            )

            metrics["reward"] += float(reward)

            metrics["cost"] += get_info_value(
                info,
                "charging_cost",
                "cost",
            )

            metrics["energy_kwh"] += energy_kwh

            metrics["grid_penalty"] += get_info_value(
                info,
                "grid_penalty",
            )

            metrics["soc_penalty"] += get_info_value(
                info,
                "soc_penalty",
            )

            metrics["departure_penalty"] += get_info_value(
                info,
                "departure_penalty",
            )

            metrics["peak_grid_kw"] = max(
                metrics["peak_grid_kw"],
                grid_usage,
            )

            metrics["overload_intervals"] += int(
                overload_kw > 1e-9
            )

            metrics["overload_kw_sum"] += overload_kw

            # Calculate actual departure SOC from the environment.
            # SOC is captured immediately after charging and before
            # the environment resets departed EVs to zero.
            departed_indices = info.get(
                "departed_ev_indices",
                [],
            )

            soc_after_charging = info.get(
                "soc_after_charging",
                [],
            )

            if len(departed_indices) > 0:
                for ev_index in departed_indices:
                    if len(soc_after_charging) > ev_index:
                        departure_soc = float(
                            soc_after_charging[ev_index]
                        )

                        metrics["departures"] += 1
                        metrics["departure_soc_sum"] += departure_soc

                        if departure_soc < 0.90:
                            metrics["departures_below_90_soc"] += 1

            metrics["steps"] += 1

            state = next_state

        # Finalize episode-level metrics.
        steps = metrics["steps"]

        metrics["overload_percentage"] = (
            100.0
            * metrics["overload_intervals"]
            / steps
            if steps > 0
            else 0.0
        )

        departures = metrics["departures"]

        metrics["average_departure_soc_percentage"] = (
            100.0
            * metrics["departure_soc_sum"]
            / departures
            if departures > 0
            else float("nan")
        )

        episode_results.append(metrics)

    return episode_results


# ==========================================================
# COMMAND-LINE INTERFACE
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare EV charging policies using reward, energy, "
            "cost, grid overload, and departure SOC metrics."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the processed dataset.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved Q-learning model.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation windows.",
    )

    parser.add_argument(
        "--episode-len",
        type=int,
        default=672,
        help="Number of 15-minute steps per window.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the detailed evaluation CSV.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------

    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")

    if args.episode_len < 1:
        parser.error("--episode-len must be at least 1.")

    data_path = args.data.expanduser().resolve()
    model_path = args.model.expanduser().resolve()

    if not data_path.exists():
        parser.error(f"Dataset not found: {data_path}")

    if not model_path.exists():
        parser.error(f"Q-learning model not found: {model_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        parser.error(f"Could not read dataset: {exc}")

    if "prices" not in df.columns:
        parser.error(
            "The processed dataset must contain a 'prices' column."
        )

    if not np.isfinite(
        df["prices"].to_numpy(dtype=float)
    ).all():
        parser.error("The prices column contains invalid values.")

    if len(df) < args.episode_len + 1:
        parser.error(
            "The dataset must contain at least episode_len + 1 rows."
        )

    price_threshold = float(df["prices"].median())

    try:
        start_indices = make_evaluation_windows(
            dataset_length=len(df),
            episode_len=args.episode_len,
            number_of_windows=args.episodes,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # ------------------------------------------------------
    # Configure policies
    # ------------------------------------------------------

    policies = [
        ("Q-learning", None),
        ("Never charge", never_charge),
        ("Maximum charge", maximum_charge),
        ("Cheap-only", cheap_only),
    ]

    # ------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------

    print("\nEV CHARGING POLICY EVALUATION")
    print("=" * 112)

    print(f"Dataset:             {data_path}")
    print(f"Q-learning model:    {model_path}")
    print(f"Evaluation windows:  {args.episodes}")
    print(f"Steps per window:    {args.episode_len}")
    print(f"Grid limit:          60 kW (default)")
    print(f"Price median:        {price_threshold:.4f}")

    print("Window start indices:", start_indices)

    print("=" * 112)

    results = {}
    all_rows = []

    for policy_name, policy in policies:

        episodes = evaluate_policy(
            policy_name=policy_name,
            policy=policy,
            data_path=data_path,
            start_indices=start_indices,
            episode_len=args.episode_len,
            price_threshold=price_threshold,
            model_path=(
                model_path
                if policy_name == "Q-learning"
                else None
            ),
        )

        # Average each numeric metric across windows.
        summary = {}

        for key in episodes[0]:
            values = np.asarray(
                [episode[key] for episode in episodes],
                dtype=float,
            )

            if np.isfinite(values).any():
                summary[key] = float(np.nanmean(values))
            else:
                summary[key] = float("nan")

        results[policy_name] = summary

        for window_number, (start_idx, episode) in enumerate(
            zip(start_indices, episodes),
            start=1,
        ):
            row = {
                "policy": policy_name,
                "window": window_number,
                "start_index": start_idx,
            }

            row.update(episode)
            all_rows.append(row)

    # ------------------------------------------------------
    # Print summary
    # ------------------------------------------------------

    print("\nAVERAGE RESULTS PER EVALUATION WINDOW")
    print("-" * 125)

    headers = [
        ("Policy", "policy", 18, 0),
        ("Reward", "reward", 12, 2),
        ("Cost", "cost", 12, 2),
        ("Energy kWh", "energy_kwh", 13, 2),
        ("Grid penalty", "grid_penalty", 14, 2),
        ("Departure pen.", "departure_penalty", 16, 2),
        ("Peak grid kW", "peak_grid_kw", 14, 2),
        ("Overload %", "overload_percentage", 12, 2),
        ("Avg departure SOC", "average_departure_soc_percentage", 20, 2),
    ]

    print(
        f"{'Policy':<18}"
        f"{'Reward':>12}"
        f"{'Cost':>12}"
        f"{'Energy kWh':>13}"
        f"{'Grid penalty':>14}"
        f"{'Departure pen.':>16}"
        f"{'Peak grid kW':>14}"
        f"{'Overload %':>12}"
        f"{'Avg dep. SOC %':>16}"
    )

    print("-" * 137)

    for policy_name, metrics in results.items():

        print(
            f"{policy_name:<18}"
            f"{metrics['reward']:>12.2f}"
            f"{metrics['cost']:>12.2f}"
            f"{metrics['energy_kwh']:>13.2f}"
            f"{metrics['grid_penalty']:>14.2f}"
            f"{metrics['departure_penalty']:>16.2f}"
            f"{metrics['peak_grid_kw']:>14.2f}"
            f"{metrics['overload_percentage']:>12.2f}"
            f"{metrics['average_departure_soc_percentage']:>16.2f}"
        )

    print("-" * 137)

    # ------------------------------------------------------
    # Save detailed results
    # ------------------------------------------------------

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nDetailed results saved to: {output_path}")

    print("\nINTERPRETATION")
    print("- Higher reward is better under the current reward definition.")
    print("- Compare charging cost alongside energy delivered.")
    print("- Lower overload percentage indicates better grid compliance.")
    print("- Average departure SOC indicates charging-service quality.")
    print(
        "- Never-charge may have a favorable reward because the current "
        "objective does not directly reward energy delivery."
    )
    print(
        "\nImportant: price units have not been confirmed. "
        "Do not interpret the cost column as a specific currency."
    )


if __name__ == "__main__":
    main()
