from pathlib import Path

import numpy as np
import pandas as pd


class EVChargingEnv:
    """
    EV charging simulation environment for Q-Learning and DQN.

    State (25 values):
        [SOC_0 ... SOC_9,
         Connected_0 ... Connected_9,
         price, load, pv, hour_sin, hour_cos]

    Action:
        Integer from 0 to 11.
        Aggregate charging power = action * 10 kW.

    Reward:
        -(charging cost + grid overload penalty + SOC penalty
          + departure penalty)

    One step represents 15 minutes.
    """

    # -------------------- Environment constants --------------------

    N_EV = 10
    DT = 0.25
    MAX_EV_KW = 11.0
    ACTION_STEP_KW = 10.0
    N_ACTIONS = 12
    STATE_SIZE = 25

    GRID_LIMIT = 60.0
    SOC_MIN = 0.30
    LAMBDA_GRID = 1.0
    LAMBDA_SOC = 100.0
    EFFICIENCY = 1.0

    # Effective battery capacities inferred from the dataset.
    # These are simulation parameters, not explicit dataset fields.
    CAPACITIES = np.array(
        [
            102.27,  # EV0
            102.27,  # EV1
            41.59,   # EV2
            41.59,   # EV3
            59.09,   # EV4
            102.27,  # EV5
            59.09,   # EV6
            59.09,   # EV7
            41.59,   # EV8
            87.50,   # EV9
        ],
        dtype=float,
    )

    def __init__(
        self,
        csv_path="data/processed/ev_charging_processed.csv",
        episode_len=None,
        seed=42,
        grid_limit=GRID_LIMIT,
        soc_min=SOC_MIN,
        lambda_grid=LAMBDA_GRID,
        lambda_soc=LAMBDA_SOC,
    ):
        """
        Load the processed dataset and initialize the environment.

        Args:
            csv_path: Path to the processed CSV.
            episode_len: Number of simulation steps per episode.
                         None means simulate the entire dataset.
            seed: Random seed for episode starting points.
            grid_limit: Assumed maximum grid draw in kW.
            soc_min: Minimum desired SOC for connected EVs.
            lambda_grid: Grid overload penalty weight.
            lambda_soc: Low-SOC penalty weight.
        """
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path.resolve()}"
            )

        self.df = pd.read_csv(self.csv_path)

        self.grid_limit = float(grid_limit)
        self.soc_min = float(soc_min)
        self.lambda_grid = float(lambda_grid)
        self.lambda_soc = float(lambda_soc)
        self.episode_len = episode_len

        if not np.isfinite(
            [
                self.grid_limit,
                self.soc_min,
                self.lambda_grid,
                self.lambda_soc,
            ]
        ).all():
            raise ValueError("Environment parameters must be finite.")

        if self.grid_limit <= 0:
            raise ValueError("grid_limit must be positive.")

        if not 0 <= self.soc_min <= 1:
            raise ValueError("soc_min must be between 0 and 1.")

        if self.lambda_grid < 0 or self.lambda_soc < 0:
            raise ValueError("Penalty weights cannot be negative.")

        self.rng = np.random.default_rng(seed)

        if len(self.df) < 2:
            raise ValueError("Dataset must contain at least two rows.")

        if episode_len is not None:
            if not isinstance(episode_len, (int, np.integer)):
                raise TypeError(
                    "episode_len must be an integer or None."
                )

            if episode_len < 1 or episode_len >= len(self.df):
                raise ValueError(
                    "episode_len must be positive and smaller than "
                    "the number of dataset rows."
                )

        # Validate required dataset columns.
        required_columns = (
            [f"soc.EV_{i}" for i in range(self.N_EV)]
            + [f"EV_{i}.connected" for i in range(self.N_EV)]
            + [
                "prices",
                "fixed_power_contribution_per_component.load",
                "fixed_power_contribution_per_component.pv",
                "hour_sin",
                "hour_cos",
            ]
        )

        missing = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {missing}"
            )

        # Connection status: rows x 10 EVs.
        self.conn = (
            self.df[
                [f"EV_{i}.connected" for i in range(self.N_EV)]
            ]
            .fillna(0)
            .to_numpy(dtype=float)
        )

        if not np.isfinite(self.conn).all():
            raise ValueError(
                "Connection data contains invalid values."
            )

        if not np.isin(self.conn, [0, 1]).all():
            raise ValueError(
                "Connection values must be 0 or 1."
            )

        self.conn = self.conn.astype(int)

        # Historical SOC initializes EVs and is used when they arrive.
        self.soc_data = self.df[
            [f"soc.EV_{i}" for i in range(self.N_EV)]
        ].to_numpy(dtype=float)

        if not np.isfinite(self.soc_data).all():
            raise ValueError(
                "SOC data contains missing or invalid values."
            )

        if np.any(
            (self.soc_data < 0) | (self.soc_data > 1)
        ):
            raise ValueError(
                "SOC values must be between 0 and 1."
            )

        self.soc_data = self.soc_data.copy()
        self.soc_data[self.conn == 0] = 0.0

        self.price = self.df["prices"].to_numpy(dtype=float)

        self.load = self.df[
            "fixed_power_contribution_per_component.load"
        ].to_numpy(dtype=float)

        self.pv = self.df[
            "fixed_power_contribution_per_component.pv"
        ].to_numpy(dtype=float)

        self.hsin = self.df["hour_sin"].to_numpy(dtype=float)
        self.hcos = self.df["hour_cos"].to_numpy(dtype=float)

        exogenous = np.column_stack(
            [
                self.price,
                self.load,
                self.pv,
                self.hsin,
                self.hcos,
            ]
        )

        if not np.isfinite(exogenous).all():
            raise ValueError(
                "Price, load, PV, or time features contain invalid values."
            )

        if (
            not np.isfinite(self.CAPACITIES).all()
            or np.any(self.CAPACITIES <= 0)
        ):
            raise ValueError(
                "Battery capacities must be finite and positive."
            )

        self.T = len(self.df)

        # Runtime state; reset() initializes these values.
        self.t = 0
        self.end = self.T - 1
        self.soc = np.zeros(self.N_EV, dtype=float)
        self._done = True

    # -------------------- State --------------------

    def _get_state(self):
        """Return the current 25-value observation."""
        state = np.concatenate(
            [
                self.soc,
                self.conn[self.t],
                np.array(
                    [
                        self.price[self.t],
                        self.load[self.t],
                        self.pv[self.t],
                        self.hsin[self.t],
                        self.hcos[self.t],
                    ]
                ),
            ]
        )

        assert state.shape == (self.STATE_SIZE,)

        return state.astype(np.float32)

    # -------------------- Charging allocation --------------------

    def allocate_power(self, total_kw):
        """
        Allocate aggregate charging power using lowest-SOC-first.

        Only connected EVs below 100% SOC are eligible.
        Each EV receives at most 11 kW.
        """
        power = np.zeros(self.N_EV, dtype=float)

        connected = self.conn[self.t] == 1
        eligible = connected & (self.soc < 1.0)

        # Stable sorting makes allocation deterministic for ties.
        indices = np.flatnonzero(eligible)
        indices = indices[
            np.argsort(self.soc[indices], kind="stable")
        ]

        remaining = min(
            float(total_kw),
            self.MAX_EV_KW * len(indices),
        )

        for i in indices:
            assigned = min(self.MAX_EV_KW, remaining)
            power[i] = assigned
            remaining -= assigned

            if remaining <= 1e-9:
                break

        return power

    # -------------------- Reset --------------------

    def reset(self, start=None):
        """
        Start a new episode.

        If start is omitted, choose a random valid starting row for
        fixed-length episodes. Full-dataset episodes start at row zero.

        Returns:
            Initial 25-value state.
        """
        if start is None:
            if self.episode_len is None:
                start = 0
            else:
                max_start = self.T - 1 - self.episode_len
                start = int(
                    self.rng.integers(0, max_start + 1)
                )

        if not isinstance(start, (int, np.integer)):
            raise TypeError("start must be an integer.")

        if start < 0 or start >= self.T - 1:
            raise ValueError(
                "start must identify a valid dataset row."
            )

        self.t = int(start)

        if self.episode_len is None:
            self.end = self.T - 1
        else:
            self.end = min(
                self.t + self.episode_len,
                self.T - 1,
            )

        # Initialize SOC using the dataset at the episode's start.
        self.soc = self.soc_data[self.t].copy()
        self.soc[self.conn[self.t] == 0] = 0.0

        self._done = False

        return self._get_state()

    # -------------------- One simulation step --------------------

    def step(self, action):
        """
        Execute one 15-minute charging decision.

        Args:
            action: Integer from 0 to 11.

        Returns:
            next_state, reward, done, info
        """
        if self._done:
            raise RuntimeError(
                "Episode has ended. Call reset() before step()."
            )

        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise TypeError(
                "Action must be an integer from 0 to 11."
            )

        if not 0 <= action < self.N_ACTIONS:
            raise ValueError(
                "Action must be between 0 and 11."
            )

        t = self.t

        # 1. Convert the discrete action to aggregate power.
        requested_kw = action * self.ACTION_STEP_KW

        # 2. Allocate power to connected EVs.
        power = self.allocate_power(requested_kw)
        actual_kw = float(power.sum())

        # 3. Update SOC after charging.
        soc_before = self.soc.copy()

        energy_added = (
            power * self.DT * self.EFFICIENCY
        )

        self.soc = np.minimum(
            1.0,
            self.soc + energy_added / self.CAPACITIES,
        )

        self.soc = np.clip(self.soc, 0.0, 1.0)

        # IMPORTANT: Capture SOC before processing departures.
        # Departing EVs will be reset to zero later in this step.
        soc_after_charging = self.soc.copy()

        # 4. Calculate grid draw.
        # Dataset convention: load is negative, PV is positive.
        grid_usage = -(
            self.load[t] - actual_kw + self.pv[t]
        )

        # 5. Calculate charging cost.
        energy_kwh = actual_kw * self.DT
        charging_cost = energy_kwh * self.price[t]

        # 6. Penalize grid draw above the configured limit.
        overload_kw = max(
            0.0,
            grid_usage - self.grid_limit,
        )

        grid_penalty = (
            self.lambda_grid * overload_kw ** 2
        )

        # 7. Penalize low SOC among EVs connected during this interval.
        connected = self.conn[t] == 1

        soc_shortfall = np.maximum(
            0.0,
            self.soc_min - self.soc[connected],
        )

        soc_penalty = (
            self.lambda_soc * np.sum(soc_shortfall ** 2)
        )

        # 8. Base reward is the negative total cost and penalties.
        reward = -(
            charging_cost
            + grid_penalty
            + soc_penalty
        )

        # 9. Advance time and process arrivals/departures.
        self.t += 1
        next_connected = self.conn[self.t]

        departed = (
            (self.conn[t] == 1)
            & (next_connected == 0)
        )

        arrived = (
            (self.conn[t] == 0)
            & (next_connected == 1)
        )

        # Record SOC at departure before resetting departed EVs.
        departure_soc = soc_after_charging[departed].copy()

        departure_shortfall = np.maximum(
            0.0,
            0.90 - departure_soc,
        )

        departure_penalty = (
            100.0 * np.sum(departure_shortfall ** 2)
        )

        # Include departure penalty in reward.
        reward -= departure_penalty

        # Reset departed EVs.
        self.soc[departed] = 0.0

        # Newly arriving EVs receive their dataset-recorded initial SOC.
        self.soc[arrived] = self.soc_data[
            self.t, arrived
        ]

        # Disconnected EVs must have zero SOC in the next state.
        self.soc[next_connected == 0] = 0.0

        done = self.t >= self.end
        self._done = done

        next_state = self._get_state()

        info = {
            "timestep": t,
            "requested_power_kw": float(requested_kw),
            "power_per_ev_kw": power.copy(),
            "actual_power_kw": actual_kw,
            "energy_kwh": float(energy_kwh),
            "grid_usage_kw": float(grid_usage),
            "grid_limit_kw": self.grid_limit,
            "overload_kw": float(overload_kw),
            "charging_cost": float(charging_cost),
            "grid_penalty": float(grid_penalty),
            "soc_penalty": float(soc_penalty),
            "reward": float(reward),
            "soc_before": soc_before,
            "soc_after_charging": soc_after_charging,
            "arrived_ev_indices": np.flatnonzero(
                arrived
            ).tolist(),
            "departed_ev_indices": np.flatnonzero(
                departed
            ).tolist(),
            "departure_soc": departure_soc.copy(),
            "departure_penalty": float(departure_penalty),
        }

        return next_state, float(reward), done, info