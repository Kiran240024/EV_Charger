from environment.ev_charging_env import EVChargingEnv

env = EVChargingEnv(
    csv_path="data/processed/ev_charging_processed.csv",
    episode_len=96,
    seed=42,
)

state = env.reset()

print("Initial state shape:", state.shape)
print("Initial SOC:", state[:10])
print("Connected EVs:", state[10:20])

# Choose action 3: request 30 kW aggregate charging power.
next_state, reward, done, info = env.step(3)

print("\nNext state shape:", next_state.shape)
print("Reward:", reward)
print("Power per EV:", info["power_per_ev_kw"])
print("Total power:", info["actual_power_kw"], "kW")
print("Energy delivered:", info["energy_kwh"], "kWh")
print("Grid usage:", info["grid_usage_kw"], "kW")
print("Charging cost:", info["charging_cost"])
print("Grid penalty:", info["grid_penalty"])
print("SOC penalty:", info["soc_penalty"])
print("Episode finished:", done)