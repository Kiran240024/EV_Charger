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
# 1. Environment-compatible constants
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
# 2. Fit discretization thresholds
# --------------------------------------------------

def fit_bins(
    env,
    n_price_levels=N_PRICE_LEVELS,
    n_hour_blocks=N_HOUR_BLOCKS,
    n_headroom_levels=N_HEADROOM_LEVELS,
):
    """
    Derive state discretization thresholds from environment data.
    """

    price_edges = np.quantile(
        env.price,
        np.linspace(0, 1, n_price_levels + 1)[1:-1],
    )

    # SOC thresholds: minimum, intermediate, and target levels.
    soc_edges = [SOC_MIN, 0.60, 0.90]

    return {
        "price_edges": [float(x) for x in price_edges],
        "soc_edges": soc_edges,
        "grid_limit": float(GRID_LIMIT_KW),
        "n_hour_blocks": int(n_hour_blocks),
        "n_headroom_levels": int(n_headroom_levels),
    }


# --------------------------------------------------
# 3. Determine Q-table dimensions
# --------------------------------------------------

def get_q_table_shape(bins):
    """
    Return Q-table dimensions, including the action dimension.
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
# 4. Convert the raw state into a discrete state
# --------------------------------------------------

def discretize_state(state, bins):
    """
    Convert a 25-feature environment observation into a
    five-dimensional tuple suitable for Q-table indexing.
    """

    state = np.asarray(state, dtype=float)

    if state.shape != (25,):
        raise ValueError(
            f"Expected a 25-feature state, got {state.shape}"
        )

    if not np.all(np.isfinite(state)):
        raise ValueError("State must contain only finite values.")

    # Extract state features.
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
# 5. Epsilon-greedy action selection
# --------------------------------------------------

def choose_action(q_table, discrete_state, epsilon, rng):
    """
    Select an action using epsilon-greedy exploration.
    """

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be between 0 and 1.")

    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))

    q_values = q_table[discrete_state]

    # Random tie-breaking avoids always choosing action zero.
    best_actions = np.flatnonzero(
        q_values == q_values.max()
    )

    return int(rng.choice(best_actions))


# --------------------------------------------------
# 6. Q-learning update
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

    if not isinstance(action, (int, np.integer)):
        raise ValueError("Action must be an integer.")

    if not 0 <= action < N_ACTIONS:
        raise ValueError(
            f"Action must be between 0 and {N_ACTIONS - 1}."
        )

    if not np.isfinite(reward):
        raise ValueError("Reward must be finite.")

    if not 0.0 < alpha <= 1.0:
        raise ValueError("Alpha must be in (0, 1].")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Gamma must be in [0, 1].")

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
# 7. Q-learning agent
# --------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning agent for EV charging optimization.
    """

    def __init__(
        self,
        env,
        alpha=DEFAULT_ALPHA,
        gamma=DEFAULT_GAMMA,
        epsilon=DEFAULT_EPSILON,
        seed=42,
    ):

        if not 0.0 < alpha <= 1.0:
            raise ValueError("Alpha must be in (0, 1].")

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Gamma must be in [0, 1].")

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1.")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        self.rng = np.random.default_rng(seed)

        self.bins = fit_bins(env)

        self.q_table = np.zeros(
            get_q_table_shape(self.bins),
            dtype=np.float64,
        )

    def get_discrete_state(self, state):
        """Convert a raw environment observation to a Q-table index."""

        return discretize_state(state, self.bins)

    def select_action(self, state, explore=True):
        """Select a valid charging action for the current state."""

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
        """
        Update the Q-table after a valid environment transition.

        Invalid actions are rejected before the Q-table is modified.
        """

        # Validate the action before it can index the Q-table.
        if not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer.")

        if not 0 <= action < N_ACTIONS:
            raise ValueError(
                f"Invalid action {action}. "
                f"Expected an integer from 0 to {N_ACTIONS - 1}."
            )

        # Validate the reward.
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite.")

        # Convert observations to discrete states.
        discrete_state = self.get_discrete_state(state)
        next_discrete_state = self.get_discrete_state(next_state)

        # Perform the Q-learning update.
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