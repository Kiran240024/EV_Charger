
"""
Tabular Q-learning agent for EV charging optimization.

State representation:
    1. Number of connected EVs
    2. Lowest SOC among connected EVs
    3. Electricity price level
    4. Time-of-day block
    5. Grid-headroom level

Action:
    Integer from 0 to 11, representing 0 to 110 kW
    in increments of 10 kW.
"""

import numpy as np


# --------------------------------------------------
# Environment-compatible constants
# --------------------------------------------------

N_EV = 10
N_ACTIONS = 12
ACTION_STEP_KW = 10.0

SOC_MIN = 0.30
GRID_LIMIT_KW = 60.0

# State discretization settings
N_PRICE_LEVELS = 3
N_HOUR_BLOCKS = 6
N_HEADROOM_LEVELS = 4

# Q-learning hyperparameters
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.95
DEFAULT_EPSILON = 1.0


# --------------------------------------------------
# 1. Fit discretization thresholds
# --------------------------------------------------

def fit_bins(env, n_price_levels=N_PRICE_LEVELS,
             n_hour_blocks=N_HOUR_BLOCKS,
             n_headroom_levels=N_HEADROOM_LEVELS):
    """
    Derive discretization thresholds from the environment's data.
    """

    price_edges = np.quantile(
        env.price,
        np.linspace(0, 1, n_price_levels + 1)[1:-1]
    )

    # Preserve the previous SOC-threshold design.
    # 0.30 = minimum SOC threshold; 0.90 = reference target.
    soc_edges = [SOC_MIN, 0.60, 0.90]

    return {
        "price_edges": [float(x) for x in price_edges],
        "soc_edges": soc_edges,
        "grid_limit": float(GRID_LIMIT_KW),
        "n_hour_blocks": int(n_hour_blocks),
        "n_headroom_levels": int(n_headroom_levels),
    }


# --------------------------------------------------
# 2. Determine Q-table dimensions
# --------------------------------------------------

def get_q_table_shape(bins):
    """
    Returns the dimensions of the Q-table, including actions.
    """

    return (
        N_EV + 1,
        len(bins["soc_edges"]) + 1,
        len(bins["price_edges"]) + 1,
        bins["n_hour_blocks"],
        bins["n_headroom_levels"],
        N_ACTIONS,
    )


# --------------------------------------------------
# 3. Convert the raw state into a discrete state
# --------------------------------------------------

def discretize_state(state, bins):
    """
    Convert a 25-feature environment state into a
    five-dimensional tuple for indexing the Q-table.
    """

    state = np.asarray(state, dtype=float)

    if state.shape != (25,):
        raise ValueError(
            f"Expected a 25-feature state, got {state.shape}"
        )

    # Extract the state features.
    soc = state[:N_EV]
    connected = state[N_EV:2 * N_EV]

    price, load, pv, hour_sin, hour_cos = state[20:25]

    # Feature 1: Number of connected EVs.
    connected_mask = connected > 0.5
    n_connected = int(connected_mask.sum())

    # Feature 2: Lowest SOC among connected EVs.
    if n_connected > 0:
        lowest_soc = float(soc[connected_mask].min())
    else:
        lowest_soc = 1.0

    soc_level = int(
        np.digitize(lowest_soc, bins["soc_edges"])
    )

    # Feature 3: Electricity price level.
    price_level = int(
        np.digitize(price, bins["price_edges"])
    )

    # Feature 4: Time-of-day block.
    hour = (
        np.arctan2(hour_sin, hour_cos) % (2 * np.pi)
    ) / (2 * np.pi) * 24

    hour_block = min(
        int(hour / (24 / bins["n_hour_blocks"])),
        bins["n_hour_blocks"] - 1,
    )

    # Feature 5: Grid-headroom level.
    # Load is negative and PV is positive in this dataset.
    headroom_kw = bins["grid_limit"] + load + pv

    headroom_width = (
        bins["grid_limit"] / bins["n_headroom_levels"]
    )

    headroom_level = int(
        np.clip(
            headroom_kw // headroom_width,
            0,
            bins["n_headroom_levels"] - 1,
        )
    )

    return (
        n_connected,
        soc_level,
        price_level,
        hour_block,
        headroom_level,
    )


# --------------------------------------------------
# 4. Epsilon-greedy action selection
# --------------------------------------------------

def choose_action(q_table, discrete_state, epsilon, rng):
    """
    With probability epsilon, explore a random action.
    Otherwise, exploit the best-known action.
    """

    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))

    q_values = q_table[discrete_state]

    # Random tie-breaking prevents always favoring action 0.
    best_actions = np.flatnonzero(
        q_values == q_values.max()
    )

    return int(rng.choice(best_actions))


# --------------------------------------------------
# 5. Q-learning update
# --------------------------------------------------

def update_q_table(
    q_table,
    state,
    action,
    reward,
    next_state,
    done,
    alpha=DEFAULT_ALPHA,
    gamma=DEFAULT_GAMMA,
):
    """
    Update Q(s, a) using the standard Q-learning equation.
    """

    current_q = q_table[state + (action,)]

    if done:
        target = reward
    else:
        best_future_q = np.max(q_table[next_state])
        target = reward + gamma * best_future_q

    q_table[state + (action,)] = (
        current_q + alpha * (target - current_q)
    )


# --------------------------------------------------
# 6. Q-learning agent
# --------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning agent.

    The training loop will be implemented separately in
    training/train_q_learning.py.
    """

    def __init__(
        self,
        env,
        alpha=DEFAULT_ALPHA,
        gamma=DEFAULT_GAMMA,
        epsilon=DEFAULT_EPSILON,
        seed=42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.rng = np.random.default_rng(seed)

        self.bins = fit_bins(env)

        self.q_table = np.zeros(
            get_q_table_shape(self.bins),
            dtype=np.float64,
        )

    def get_discrete_state(self, state):
        """Convert a raw environment state to a Q-table index."""
        return discretize_state(state, self.bins)

    def select_action(self, state, explore=True):
        """Select an action for the current raw state."""

        discrete_state = self.get_discrete_state(state)

        epsilon = self.epsilon if explore else 0.0

        return choose_action(
            self.q_table,
            discrete_state,
            epsilon,
            self.rng,
        )

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """Update the Q-table after one environment transition."""

        discrete_state = self.get_discrete_state(state)
        next_discrete_state = self.get_discrete_state(next_state)

        update_q_table(
            self.q_table,
            discrete_state,
            action,
            reward,
            next_discrete_state,
            done,
            alpha=self.alpha,
            gamma=self.gamma,
        )
