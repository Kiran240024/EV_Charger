
from environment.ev_charging_env import EVChargingEnv
from agents.q_learning import QLearningAgent

env = EVChargingEnv(
    csv_path="data/processed/ev_charging_processed.csv",
    episode_len=96,
    seed=42,
)

agent = QLearningAgent(env, seed=42)

state = env.reset()

print("Q-table shape:", agent.q_table.shape)
print("Discrete state:", agent.get_discrete_state(state))

action = agent.select_action(state)
next_state, reward, done, info = env.step(action)

agent.learn(state, action, reward, next_state, done)

print("Selected action:", action)
print("Requested power:", action * 10, "kW")
print("Reward:", reward)
print("Q-table update completed!")
