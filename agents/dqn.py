"""
Deep Q-Network agent for EV Charging Optimization.

State:
    25 features representing EV, electricity, and grid conditions.

Actions:
    12 discrete actions representing 0 to 110 kW.

Components:
    - Online neural network
    - Target neural network
    - Experience replay
    - Epsilon-greedy exploration
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------
# Configuration
# --------------------------------------------------

STATE_SIZE = 25
N_ACTIONS = 12

HIDDEN_SIZE = 128

DEFAULT_GAMMA = 0.95
DEFAULT_LR = 0.001
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_MIN = 0.05
DEFAULT_EPSILON_DECAY = 0.995

DEFAULT_BATCH_SIZE = 64
DEFAULT_MEMORY_SIZE = 10000
DEFAULT_TARGET_UPDATE = 100


# --------------------------------------------------
# Neural network
# --------------------------------------------------

class DQNNetwork(nn.Module):
    """
    Maps a 25-feature state to 12 action-value predictions.
    """

    def __init__(self, state_size=STATE_SIZE,
                 n_actions=N_ACTIONS):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE),
            nn.ReLU(),

            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),

            nn.Linear(HIDDEN_SIZE, n_actions),
        )

    def forward(self, state):
        return self.network(state)


# --------------------------------------------------
# Deep Q-Network agent
# --------------------------------------------------

class DQNAgent:

    def __init__(
        self,
        state_size=STATE_SIZE,
        n_actions=N_ACTIONS,
        learning_rate=DEFAULT_LR,
        gamma=DEFAULT_GAMMA,
        epsilon=DEFAULT_EPSILON,
        epsilon_min=DEFAULT_EPSILON_MIN,
        epsilon_decay=DEFAULT_EPSILON_DECAY,
        batch_size=DEFAULT_BATCH_SIZE,
        memory_size=DEFAULT_MEMORY_SIZE,
        target_update=DEFAULT_TARGET_UPDATE,
        seed=42,
    ):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.n_actions = n_actions

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cpu")

        # Online network learns from experience.
        self.policy_net = DQNNetwork(
            state_size, n_actions
        ).to(self.device)

        # Target network stabilizes learning.
        self.target_net = DQNNetwork(
            state_size, n_actions
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()

        # Experience replay memory.
        self.memory = deque(maxlen=memory_size)

        self.steps = 0

    # --------------------------------------------------
    # Action selection
    # --------------------------------------------------

    def select_action(self, state, explore=True):
        """
        Epsilon-greedy action selection.

        Returns an integer between 0 and 11.
        """

        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    # --------------------------------------------------
    # Store experience
    # --------------------------------------------------

    def remember(
        self, state, action, reward, next_state, done
    ):
        """
        Store one transition in replay memory.
        """

        self.memory.append((
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            float(done),
        ))

    # --------------------------------------------------
    # Train from replay memory
    # --------------------------------------------------

    def learn(self):
        """
        Perform one DQN optimization step.

        Returns the loss, or None if memory is too small.
        """

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(
            self.memory, self.batch_size
        )

        states = torch.as_tensor(
            np.array([x[0] for x in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            [x[1] for x in batch],
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            [x[2] for x in batch],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.array([x[3] for x in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            [x[4] for x in batch],
            dtype=torch.float32,
            device=self.device,
        )

        # Q(s, a) predicted by the online network.
        current_q = self.policy_net(
            states
        ).gather(1, actions).squeeze(1)

        # Bellman target calculated using the target network.
        with torch.no_grad():
            next_q = self.target_net(
                next_states
            ).max(dim=1)[0]

            target_q = rewards + (
                self.gamma * next_q * (1.0 - dones)
            )

        loss = self.loss_function(
            current_q, target_q
        )

        self.optimizer.zero_grad()
        loss.backward()

        # Prevent excessively large gradients.
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=10.0
        )

        self.optimizer.step()

        # Periodically synchronize the target network.
        self.steps += 1

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return float(loss.item())

    # --------------------------------------------------
    # Exploration decay
    # --------------------------------------------------

    def decay_epsilon(self):
        """
        Gradually reduce random exploration.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    # --------------------------------------------------
    # Save and load models
    # --------------------------------------------------

    def save(self, path):
        """
        Save the online network and optimizer state.
        """

        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }, path)

    def load(self, path):
        """
        Restore a previously saved model.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]

        self.target_net.eval()