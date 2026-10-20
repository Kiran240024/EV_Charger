# ⚡ EVora — Reinforcement Learning for Smart EV Charging

EVora is a research project exploring reinforcement learning for intelligent
electric vehicle charging. It combines a simulated multi-EV charging
environment, Q-learning, baseline charging strategies, and a Streamlit
dashboard for training and evaluating charging policies.

The objective is to study how charging decisions affect electricity costs,
energy delivery, grid loading, and battery state of charge (SOC).

## ✨ Features

- **EV charging simulation:** Simulates multiple electric vehicles connected
  to a shared charging system.
- **Q-learning:** Trains an agent to select charging actions.
- **DQN implementation:** Includes a Deep Q-Network implementation.
- **Baseline comparisons:** Evaluates learned policies against simpler
  charging strategies.
- **Interactive dashboard:** Uses Streamlit to access training, evaluation,
  and saved results.
- **Performance metrics:** Examines charging cost, energy delivery,
  grid overload, and departure SOC.
- **Automated tests:** Tests the environment and Q-learning components.

## 🖥️ Dashboard

The EVora dashboard provides four main sections:

| Section | Purpose |
| --- | --- |
| Overview | Inspect dataset and saved model availability |
| Train model | Configure and run Q-learning training |
| Evaluate agent | Evaluate a saved model against baseline strategies |
| Results | Explore evaluation CSV files and visualise numeric metrics |

## 📁 Project Structure

```text
EV_Charger/
├── agents/
│   ├── q_learning.py
│   └── dqn.py
├── data/
│   ├── raw/
│   └── processed/
├── environment/
│   └── ev_charging_env.py
├── evaluation/
│   ├── baselines.py
│   ├── evaluate.py
│   └── evaluate_dqn.py
├── models/
├── testing/
│   ├── test_environment.py
│   └── test_q_learning.py
├── training/
│   ├── train_q_learning.py
│   └── train_dqn.py
├── app.py
├── preprocess.py
├── requirements.txt
└── README.md
```

## 🛠️ Requirements

- Python 3.9 or later, subject to dependency compatibility
- pip
- The processed EV charging dataset
- Streamlit for the dashboard

## 🚀 Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/Kiran240024/EV_Charger.git
cd EV_Charger
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Streamlit if it is not already included in the requirements:

```bash
python -m pip install streamlit
```

## 📊 Dataset

The training script expects the processed dataset at:

```text
data/processed/ev_charging_processed.csv
```

Make sure the dataset is available at this location before training.

## ▶️ Launch the Dashboard

From the project root, run:

```bash
python -m streamlit run app.py
```

Open the local URL printed in your terminal, usually:

```text
http://localhost:8501
```

## 🧠 Train a Q-Learning Agent

Run a short training experiment:

```bash
python -m training.train_q_learning \
  --episodes 10 \
  --episode-len 96 \
  --output models/q_learning_smoke_test.npz
```

The model is saved in the `models/` directory.

The dashboard also provides a form for configuring training parameters and
choosing an output filename.

A short training run verifies that the workflow executes; it does not
establish that the policy has converged or learned an optimal strategy.

## 📈 Evaluate a Trained Model

For example, evaluate a saved 200-episode model:

```bash
python -m evaluation.evaluate \
  --model models/q_learning_model_200.npz \
  --episodes 5 \
  --episode-len 96 \
  --output evaluation/evaluation_results_200.csv
```

Ensure the specified model file exists before running the command.

The evaluation module compares the learned policy with the baseline strategies
implemented in the project.

## 🧪 Run the Tests

Install the dependencies first, then run:

```bash
python -m pytest testing/ -v
```

The tests cover environment behaviour and Q-learning functionality.

## 📌 Evaluation Metrics

Depending on the evaluation output, the project examines:

- Cumulative reward
- Charging cost
- Energy delivered
- Grid overload penalties
- Peak grid demand
- Number and percentage of overload intervals
- Departure state of charge

Interpret reward alongside the underlying metrics. A higher or lower reward
alone does not necessarily establish that a charging policy is better.

## 🔬 Project Scope

This project focuses on simulation and reinforcement-learning-based charging
decisions. Results depend on the dataset, environment configuration, reward
function, training parameters, and evaluation windows.

The dashboard visualises saved experiment outputs; it does not represent
live charging-station telemetry.

## 👩‍💻 Repository

GitHub: https://github.com/Kiran240024/EV_Charger