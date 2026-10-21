"""
Compare Immediate Charging, Price-Based Charging,
Q-Learning, and DQN on identical EV charging windows.

Run:
    python -m evaluation.compare_strategies

Outputs:
    evaluation/comparison_summary.csv
    evaluation/comparison_details.csv
    evaluation/plots/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.dqn import DQNAgent
from agents.q_learning import QLearningAgent
from environment.ev_charging_env import EVChargingEnv


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/ev_charging_processed.csv"
Q_MODEL_PATH = ROOT / "models/q_learning_model.npz"
DQN_MODEL_PATH = ROOT / "models/dqn_model.pth"

OUTPUT_DIR = ROOT / "evaluation"
PLOT_DIR = OUTPUT_DIR / "plots"

EPISODES = 5
EPISODE_LENGTH = 96
TARGET_SOC = 0.90
DT_HOURS = 0.25


def load_q_agent(env):
    """Load the existing Q-learning table and its saved bins."""
    if not Q_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing Q-learning model: {Q_MODEL_PATH}")

    with np.load(Q_MODEL_PATH, allow_pickle=False) as saved:
        q_table = saved["Q"].copy()
        bins = json.loads(str(saved["bins"].item()))

    agent = QLearningAgent(env, epsilon=0.0, seed=42)

    if q_table.shape != agent.q_table.shape:
        raise ValueError(
            f"Q-table shape mismatch: saved {q_table.shape}, "
            f"expected {agent.q_table.shape}"
        )

    agent.q_table = q_table
    agent.bins = bins
    agent.epsilon = 0.0
    return agent


def create_windows(dataset_length):
    """Choose reproducible evaluation windows shared by every strategy."""
    if dataset_length <= EPISODE_LENGTH:
        raise ValueError("Dataset is too short for the chosen episode length.")

    max_start = dataset_length - 1 - EPISODE_LENGTH
    starts = np.linspace(0, max_start, EPISODES, dtype=int).tolist()

    if len(set(starts)) != len(starts):
        raise ValueError("Could not create distinct evaluation windows.")

    return starts


def choose_action(strategy, state, q_agent, dqn_agent, price_threshold):
    """Return the action for the selected strategy."""
    connected = state[10:20] > 0.5

    if strategy == "Immediate Charging":
        return 11 if connected.any() else 0

    if strategy == "Price-Based Charging":
        return 11 if connected.any() and state[20] <= price_threshold else 0

    if strategy == "Q-Learning":
        return q_agent.select_action(state, explore=False)

    if strategy == "DQN":
        return dqn_agent.select_action(state, explore=False)

    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate(strategy, start, price_threshold):
    """Evaluate one strategy on one fixed window."""
    env = EVChargingEnv(
        csv_path=str(DATA_PATH),
        episode_len=EPISODE_LENGTH,
        seed=42,
    )

    q_agent = load_q_agent(env) if strategy == "Q-Learning" else None

    dqn_agent = None
    if strategy == "DQN":
        dqn_agent = DQNAgent(
            state_size=env.STATE_SIZE,
            n_actions=env.N_ACTIONS,
            seed=42,
        )
        dqn_agent.load(str(DQN_MODEL_PATH))
        dqn_agent.epsilon = 0.0
        dqn_agent.policy_net.eval()

    state = env.reset(start=int(start))

    cost = 0.0
    energy = 0.0
    reward_total = 0.0
    peak_grid = float("-inf")

    # Arrival-to-first-charge delay tracking, measured in 15-minute steps.
    active_sessions = {}
    session_delays_hours = []
    session_never_charged = 0

    departures = 0
    deadline_violations = 0
    departure_soc_sum = 0.0
    successful_departures = 0

    time_rows = []
    step_number = 0
    done = False

    # EVs already connected at the start are treated as sessions
    # beginning at the start of this evaluation window.
    for ev_idx in np.flatnonzero(env.conn[env.t] == 1):
        active_sessions[int(ev_idx)] = {
            "arrival_step": 0,
            "charged": False,
        }

    while not done:
        action = choose_action(
            strategy, state, q_agent, dqn_agent, price_threshold
        )

        next_state, reward, done, info = env.step(action)

        cost += float(info["charging_cost"])
        energy += float(info["energy_kwh"])
        reward_total += float(reward)
        peak_grid = max(peak_grid, float(info["grid_usage_kw"]))

        powers = np.asarray(info["power_per_ev_kw"], dtype=float)
        soc_before = np.asarray(info["soc_before"], dtype=float)
        soc_after = np.asarray(info["soc_after_charging"], dtype=float)

        # Count a session's delay until it first receives actual power.
        for ev_idx, session in list(active_sessions.items()):
            if not session["charged"] and powers[ev_idx] > 1e-9:
                session["charged"] = True
                session_delays_hours.append(
                    (step_number - session["arrival_step"]) * DT_HOURS
                )

        # Record completed sessions before the environment resets SOC.
        for ev_idx in info["departed_ev_indices"]:
            ev_idx = int(ev_idx)
            departures += 1
            departure_soc = float(soc_after[ev_idx])
            departure_soc_sum += departure_soc

            if departure_soc >= TARGET_SOC:
                successful_departures += 1
            else:
                deadline_violations += 1

            session = active_sessions.pop(ev_idx, None)
            if session is not None and not session["charged"]:
                session_never_charged += 1

        # The next state already contains the arrival SOCs.
        for ev_idx in info["arrived_ev_indices"]:
            active_sessions[int(ev_idx)] = {
                "arrival_step": step_number + 1,
                "charged": False,
            }

        time_rows.append({
            "strategy": strategy,
            "window_start": int(start),
            "step": step_number,
            "time_hours": step_number * DT_HOURS,
            "cost": float(info["charging_cost"]),
            "energy_kwh": float(info["energy_kwh"]),
            "grid_load_kw": float(info["grid_usage_kw"]),
            "mean_connected_soc": (
                float(np.mean(next_state[:10][next_state[10:20] > 0.5]))
                if np.any(next_state[10:20] > 0.5)
                else np.nan
            ),
            "reward": float(reward),
        })

        state = next_state
        step_number += 1

    # Measure final SOC only for EVs connected at the final state.
    final_connected = state[10:20] > 0.5
    final_soc_values = state[:10][final_connected]
    final_soc = (
        float(np.mean(final_soc_values))
        if len(final_soc_values)
        else np.nan
    )

    # Include still-active sessions that never charged in a separate
    # count; they are not counted as completed departures.
    session_never_charged += sum(
        1 for session in active_sessions.values() if not session["charged"]
    )

    metrics = {
        "strategy": strategy,
        "window_start": int(start),
        "total_charging_cost": cost,
        "total_energy_kwh": energy,
        "peak_grid_load_kw": peak_grid,
        "final_mean_soc_percent": final_soc * 100 if np.isfinite(final_soc) else np.nan,
        "average_charging_delay_minutes": (
            float(np.mean(session_delays_hours) * 60)
            if session_delays_hours else np.nan
        ),
        "deadline_violations": deadline_violations,
        "completed_sessions": departures,
        "charging_success_rate_percent": (
            100.0 * successful_departures / departures
            if departures else np.nan
        ),
        "average_departure_soc_percent": (
            100.0 * departure_soc_sum / departures
            if departures else np.nan
        ),
        "sessions_never_charged": session_never_charged,
        "total_reward": reward_total,
        "steps": step_number,
    }

    return metrics, time_rows


def save_plots(details, summary):
    """Generate the four-strategy comparison charts."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charging cost comparison.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["strategy"], summary["total_charging_cost"])
    ax.set_title("Total Charging Cost by Strategy")
    ax.set_ylabel("Charging cost")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cost_comparison.png", dpi=160)
    plt.close(fig)

    # 2. Energy consumption comparison.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["strategy"], summary["total_energy_kwh"])
    ax.set_title("Total Energy Consumed by Strategy")
    ax.set_ylabel("Energy (kWh)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "energy_comparison.png", dpi=160)
    plt.close(fig)

    # 3. Mean connected SOC over time, using the first evaluation window.
    first_window = details[details["window_start"] == details["window_start"].min()]
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, group in first_window.groupby("strategy"):
        ax.plot(
            group["time_hours"],
            group["mean_connected_soc"] * 100,
            label=strategy,
        )
    ax.set_title("Mean Connected EV SOC Over Time")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Mean SOC (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "soc_over_time.png", dpi=160)
    plt.close(fig)

    # 4. Grid load over time.
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, group in first_window.groupby("strategy"):
        ax.plot(
            group["time_hours"],
            group["grid_load_kw"],
            label=strategy,
        )
    ax.set_title("Grid Load Over Time")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Grid load (kW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "grid_load_over_time.png", dpi=160)
    plt.close(fig)

    # 5. DQN training reward from the recorded training CSV, if available.
    training_csv = ROOT / "training" / "dqn_training_rewards.csv"
    if training_csv.exists():
        training = pd.read_csv(training_csv)
        reward_column = next(
            (col for col in ["reward", "total_reward", "episode_reward"]
             if col in training.columns),
            None,
        )
        if reward_column:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                np.arange(1, len(training) + 1),
                training[reward_column],
                marker="o",
            )
            ax.set_title("DQN Training Reward vs Episodes")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total reward")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / "dqn_training_reward.png", dpi=160)
            plt.close(fig)


def main():
    for required in [DATA_PATH, Q_MODEL_PATH, DQN_MODEL_PATH]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    data = pd.read_csv(DATA_PATH)
    if "prices" not in data.columns:
        raise ValueError("Dataset must contain a 'prices' column.")

    price_threshold = float(data["prices"].median())
    starts = create_windows(len(data))

    strategies = [
        "Immediate Charging",
        "Price-Based Charging",
        "Q-Learning",
        "DQN",
    ]

    print("\nEV CHARGING STRATEGY COMPARISON")
    print("=" * 72)
    print(f"Dataset: {DATA_PATH}")
    print(f"Evaluation windows: {len(starts)}")
    print(f"Steps per window: {EPISODE_LENGTH}")
    print(f"Price threshold (dataset median): {price_threshold:.4f}")
    print(f"Window start indices: {starts}")

    all_metrics = []
    all_time_rows = []

    for strategy in strategies:
        print(f"\nEvaluating {strategy}...")
        for window_number, start in enumerate(starts, start=1):
            metrics, time_rows = evaluate(strategy, start, price_threshold)
            metrics["window"] = window_number
            all_metrics.append(metrics)
            all_time_rows.extend(time_rows)

            print(
                f"  Window {window_number}/{len(starts)} | "
                f"Cost={metrics['total_charging_cost']:.2f} | "
                f"Energy={metrics['total_energy_kwh']:.2f} kWh | "
                f"Peak grid={metrics['peak_grid_load_kw']:.2f} kW"
            )

    details = pd.DataFrame(all_metrics)
    timeseries = pd.DataFrame(all_time_rows)

    # Summary is the average across identical windows.
    numeric_columns = [
        column for column in details.select_dtypes(include="number").columns
        if column not in {"window_start", "window"}
    ]
    summary = (
        details.groupby("strategy", as_index=False)[numeric_columns]
        .mean(numeric_only=True)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details.to_csv(OUTPUT_DIR / "comparison_details.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)
    timeseries.to_csv(OUTPUT_DIR / "comparison_timeseries.csv", index=False)

    save_plots(timeseries, summary)

    print("\nSUMMARY (AVERAGE PER EVALUATION WINDOW)")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'comparison_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'comparison_details.csv'}")
    print(f"  {OUTPUT_DIR / 'comparison_timeseries.csv'}")
    print(f"  {PLOT_DIR}")


if __name__ == "__main__":
    main()
