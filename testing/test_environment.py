import numpy as np
import pytest

from environment.ev_charging_env import EVChargingEnv


DATA_PATH = "data/processed/ev_charging_processed.csv"


@pytest.fixture
def env():
    """Create a fresh EV charging environment for each test."""
    return EVChargingEnv(
        csv_path=DATA_PATH,
        episode_len=96,
        seed=42,
    )


def test_environment_initializes(env):
    """The environment should initialize with a valid state."""
    state = env.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (25,)
    assert np.all(np.isfinite(state))


def test_reset_returns_valid_state(env):
    """Reset should return a valid observation."""
    state = env.reset()

    assert state.shape == (25,)
    assert np.all(np.isfinite(state))


def test_step_returns_expected_values(env):
    """A valid action should produce a complete transition."""
    env.reset()

    next_state, reward, done, info = env.step(3)

    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (25,)
    assert np.all(np.isfinite(next_state))

    assert isinstance(reward, (int, float, np.number))
    assert np.isfinite(reward)

    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_step_reports_expected_metrics(env):
    """The transition should report the metrics used for evaluation."""
    env.reset()

    _, _, _, info = env.step(3)

    expected_keys = {
        "energy_kwh",
        "charging_cost",
        "grid_penalty",
        "soc_penalty",
        "departure_penalty",
        "grid_usage_kw",
    }

    assert expected_keys.issubset(info.keys())

    for key in expected_keys:
        assert np.isfinite(float(info[key]))


@pytest.mark.parametrize("action", [-1, 12])
def test_invalid_action_is_rejected(env, action):
    """Actions outside the documented range should be rejected."""
    env.reset()

    with pytest.raises((ValueError, IndexError)):
        env.step(action)


def test_episode_finishes_after_configured_steps(env):
    """The episode should terminate at its configured length."""
    env.reset()

    done = False
    steps = 0

    while not done and steps < 100:
        _, _, done, _ = env.step(0)
        steps += 1

    assert done
    assert steps == 96