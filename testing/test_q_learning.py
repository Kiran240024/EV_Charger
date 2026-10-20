import numpy as np
import pytest

from environment.ev_charging_env import EVChargingEnv
from agents.q_learning import QLearningAgent


DATA_PATH = "data/processed/ev_charging_processed.csv"


@pytest.fixture
def setup_agent():
    """Create a fresh environment and Q-learning agent."""
    env = EVChargingEnv(
        csv_path=DATA_PATH,
        episode_len=96,
        seed=42,
    )

    agent = QLearningAgent(env, seed=42)
    state = env.reset()

    return env, agent, state


def test_q_table_has_valid_shape(setup_agent):
    """The Q-table should have state dimensions and 12 actions."""
    _, agent, _ = setup_agent

    assert agent.q_table.ndim == 6
    assert agent.q_table.shape[-1] == 12
    assert np.all(np.isfinite(agent.q_table))


def test_state_discretization(setup_agent):
    """The continuous observation should map to a valid discrete state."""
    _, agent, state = setup_agent

    discrete_state = agent.get_discrete_state(state)

    assert isinstance(discrete_state, tuple)
    assert len(discrete_state) == 5

    for index, size in zip(discrete_state, agent.q_table.shape[:-1]):
        assert isinstance(index, (int, np.integer))
        assert 0 <= index < size


def test_action_is_within_valid_range(setup_agent):
    """The agent should choose an action between 0 and 11."""
    _, agent, state = setup_agent

    action = agent.select_action(state)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 12


def test_environment_transition_and_learning(setup_agent):
    """The agent should learn from an environment transition."""
    env, agent, state = setup_agent

    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)

    assert next_state.shape == (25,)
    assert np.isfinite(reward)
    assert isinstance(info, dict)

    agent.learn(state, action, reward, next_state, done)

    assert np.all(np.isfinite(agent.q_table))


def test_learning_updates_q_table(setup_agent):
    """A non-terminal transition should update the selected Q-value."""
    _, agent, state = setup_agent

    discrete_state = agent.get_discrete_state(state)
    action = 0

    # Use a zero reward and a zero future Q-value initially.
    # Set a known nonzero reward to make the expected update observable.
    reward = 10.0
    next_discrete_state = discrete_state

    old_value = agent.q_table[discrete_state + (action,)]

    agent.learn(
        state=state,
        action=action,
        reward=reward,
        next_state=state,
        done=True,
    )

    new_value = agent.q_table[discrete_state + (action,)]

    assert new_value != old_value
    assert np.isfinite(new_value)


@pytest.mark.parametrize("action", [-1, 12])
def test_agent_learning_rejects_invalid_actions(setup_agent, action):
    """Invalid action indices should not be accepted for learning."""
    _, agent, state = setup_agent

    with pytest.raises((ValueError, IndexError)):
        agent.learn(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=True,
        )