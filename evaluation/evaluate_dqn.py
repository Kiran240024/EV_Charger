"""
Evaluate a trained DQN agent and calculate all seven
EV-charging performance metrics.

Outputs:
    evaluation/dqn_evaluation_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from agents.dqn import DQNAgent
from environment.ev_charging_env import EVChargingEnv


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "data/processed/ev_charging_processed.csv"
MODEL_PATH = ROOT / "models/dqn_model.pth"
OUTPUT_PATH = ROOT / "evaluation/dqn_evaluation_results.csv"

EPISODE_LENGTH = 96
DEPARTURE_SOC_TARGET = 0.90


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DQN checkpoint not found: {MODEL_PATH}"
        )

    env = EVChargingEnv(
        csv_path=str(DATASET_PATH),
        episode_len=EPISODE_LENGTH,
        seed=100,
    )

    agent = DQNAgent(
        state_size=env.STATE_SIZE,
        n_actions=env.N_ACTIONS,
        seed=42,
    )

    agent.load(str(MODEL_PATH))
    agent.epsilon = 0.0

    state = env.reset()

    total_reward = 0.0
    total_cost = 0.0
    total_energy = 0.0

    peak_grid_load = float("-inf")
    total_overload = 0.0

    action_counts = np.zeros(env.N_ACTIONS, dtype=int)

    # Track charging delays and departure outcomes by EV index.
    first_charge_step = {}
    arrival_step = {}
    departure_socs = []
    deadline_violations = 0
    completed_departures = 0

    # The dataset may already contain connected EVs at reset.
    for ev_index in range(env.N_EV):
        if env.conn[env.t, ev_index] == 1:
            arrival_step[ev_index] = 0

    step_number = 0
    done = False

    while not done:
        action = agent.select_action(state, explore=False)
        action_counts[action] += 1

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_cost += info["charging_cost"]
        total_energy += info["energy_kwh"]

        peak_grid_load = max(
            peak_grid_load,
            info["grid_usage_kw"],
        )

        total_overload += info["overload_kw"]

        # Record the first interval in which each EV receives power.
        for ev_index, power_kw in enumerate(info["power_per_ev_kw"]):
            if power_kw > 0 and ev_index not in first_charge_step:
                first_charge_step[ev_index] = step_number

        # Register newly arriving EVs.
        for ev_index in info["arrived_ev_indices"]:
            arrival_step[ev_index] = step_number + 1

        # Evaluate EVs leaving the charging station.
        for soc in info["departure_soc"]:
            departure_soc = float(soc)
            departure_socs.append(departure_soc)
            completed_departures += 1

            if departure_soc < DEPARTURE_SOC_TARGET:
                deadline_violations += 1

        state = next_state
        step_number += 1

    # Mean SOC among EVs still connected at the end.
    connected_at_end = env.conn[env.t] == 1

    if np.any(connected_at_end):
        final_mean_soc = float(
            np.mean(env.soc[connected_at_end])
        )
    else:
        final_mean_soc = float("nan")

    # Calculate delay for EVs that received charging.
    # Each simulation interval represents 15 minutes.
    charging_delays_minutes = []

    for ev_index, first_step in first_charge_step.items():
        arrival = arrival_step.get(ev_index, 0)
        delay_steps = max(0, first_step - arrival)
        charging_delays_minutes.append(
            delay_steps * env.DT * 60
        )

    never_charged_count = len(
        set(arrival_step) - set(first_charge_step)
    )

    average_delay_minutes = (
        float(np.mean(charging_delays_minutes))
        if charging_delays_minutes
        else float("nan")
    )

    charging_success_rate = (
        100.0
        * sum(
            soc >= DEPARTURE_SOC_TARGET
            for soc in departure_socs
        )
        / completed_departures
        if completed_departures
        else float("nan")
    )

    average_departure_soc = (
        float(np.mean(departure_socs))
        if departure_socs
        else float("nan")
    )

    results = {
        "strategy": "DQN",
        "total_charging_cost": total_cost,
        "total_energy_kwh": total_energy,
        "peak_grid_load_kw": peak_grid_load,
        "final_mean_soc_percent": final_mean_soc * 100,
        "average_charging_delay_minutes": average_delay_minutes,
        "deadline_violations": deadline_violations,
        "charging_success_rate_percent": charging_success_rate,
        "average_departure_soc_percent": average_departure_soc * 100,
        "completed_departures": completed_departures,
        "never_charged_ev_count": never_charged_count,
        "total_reward": total_reward,
        "average_reward_per_step": total_reward / step_number,
        "accumulated_overload_kw": total_overload,
        "evaluation_steps": step_number,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([results]).to_csv(OUTPUT_PATH, index=False)

    print("\nDQN Evaluation Results")
    print("=" * 48)

    for metric, value in results.items():
        print(f"{metric}: {value}")

    print("\nAction distribution")
    print("-" * 48)

    for action, count in enumerate(action_counts):
        power_kw = action * env.ACTION_STEP_KW
        print(f"{power_kw:>3} kW: {count:>3} intervals")

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()