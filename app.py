from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "ev_charging_processed.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "evaluation"
TRAINING_DIR = ROOT / "training"
PLOTS_DIR = RESULTS_DIR / "plots"

SUMMARY_FILE = RESULTS_DIR / "comparison_summary.csv"
DETAILS_FILE = RESULTS_DIR / "comparison_details.csv"
TIMESERIES_FILE = RESULTS_DIR / "comparison_timeseries.csv"
DQN_REWARDS_FILE = TRAINING_DIR / "dqn_training_rewards.csv"

DEFAULT_MODEL = MODEL_DIR / "q_learning_model_200.npz"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EVora | Smart Charging",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --green: #28745B;
        --green-dark: #194D3B;
        --green-light: #E6F2EA;
        --cream: #F8F8F3;
        --ink: #202C25;
        --muted: #758078;
        --border: #E4E9E2;
        --white: #FFFFFF;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--cream);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: var(--green-dark);
        margin-bottom: 0;
    }

    .brand-dot {
        color: #D58A57;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 0.72rem;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    .eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: var(--green);
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.5rem);
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -2px;
        color: var(--ink);
        margin-bottom: 16px;
    }

    .hero-title span {
        color: var(--green);
    }

    .hero-description {
        font-size: 1rem;
        line-height: 1.8;
        color: var(--muted);
        max-width: 680px;
    }

    .hero-box {
        background: linear-gradient(125deg, #E8F2E9, #F4F1E7);
        border: 1px solid #E0E9DD;
        border-radius: 24px;
        padding: 38px;
        margin: 12px 0 28px 0;
        position: relative;
        overflow: hidden;
    }

    .hero-note {
        font-size: 0.85rem;
        color: #536A5B;
        margin-top: 20px;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 24px 0 6px 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 20px;
    }

    .metric-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        min-height: 125px;
    }

    .metric-label {
        font-size: 0.82rem;
        color: var(--muted);
        margin-bottom: 12px;
    }

    .metric-value {
        font-family: 'Manrope', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: var(--ink);
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 8px;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 24px;
        min-height: 170px;
    }

    .info-icon {
        font-size: 1.7rem;
        margin-bottom: 12px;
    }

    .info-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .info-text {
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.7;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.82rem;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 43px;
        border: 1px solid #D9E2D9;
    }

    .stButton > button[kind="primary"] {
        background: var(--green);
        color: white;
        border: 1px solid var(--green);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--green-dark);
        border-color: var(--green-dark);
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        padding: 24px;
        border-radius: 18px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    hr {
        border-color: var(--border);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .hero-box {
            padding: 24px;
        }

        .hero-title {
            letter-spacing: -1px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(icon, title, description):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-icon">{icon}</div>
            <div class="info-title">{title}</div>
            <div class="info-text">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def available_models():
    return sorted(
        MODEL_DIR.glob("*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def available_results():
    return sorted(
        RESULTS_DIR.glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_results(path):
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def run_command(command, timeout=1800):
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return 124, "", (
            f"The process exceeded the {timeout}-second time limit. "
            "Try a smaller number of episodes."
        )

    except OSError as exc:
        return 1, "", str(exc)


def show_command_result(return_code, stdout, stderr):
    if stdout.strip():
        st.markdown("#### Run log")
        st.code(stdout[-15000:], language="text")

    if return_code == 0:
        st.success("Operation completed successfully.", icon="✅")
    else:
        st.error("The operation did not complete successfully.", icon="❌")

        if stderr.strip():
            st.markdown("#### Error details")
            st.code(stderr[-12000:], language="text")

        st.info(
            "Check that your dataset, project dependencies, and Python "
            "modules are available."
        )


def show_page_heading(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <div class="section-title" style="font-size:2rem;margin-top:0">
            {title}
        </div>
        <div class="hero-description">{description}</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def download_csv_button(path, label, key):
    if path.exists():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            key=key,
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">EVora<span class="brand-dot">.</span></div>
        <div class="brand-caption">Smart charging intelligence</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.divider()
    st.markdown("##### WORKSPACE")

    pages = [
        "Overview",
        "Train model",
        "Evaluate agent",
        "Strategy Comparison",
        "Results",
    ]

    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"

    if "requested_page" in st.session_state:
        requested = st.session_state.pop("requested_page")
        if requested in pages:
            st.session_state["page"] = requested

    selected_page = st.radio(
        "Navigation",
        pages,
        key="page",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("##### PROJECT STATUS")

    if DATA_FILE.exists():
        st.success("Dataset available", icon="✅")
    else:
        st.error("Dataset missing", icon="❌")

    st.caption(f"Saved Q-learning models · {len(available_models())}")
    st.caption(f"Evaluation files · {len(available_results())}")

    st.divider()
    st.markdown(
        """
        <div class="small-muted">
            EVora · Reinforcement learning<br>
            Intelligent EV charging research
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if selected_page == "Overview":

    st.markdown(
        """
        <div class="hero-box">
            <div class="eyebrow">REINFORCEMENT LEARNING · EV ENERGY</div>
            <div class="hero-title">
                Charging smarter.<br>
                <span>Wasting less.</span>
            </div>
            <div class="hero-description">
                Explore how intelligent charging policies balance
                energy costs, grid demand and battery requirements.
            </div>
            <div class="hero-note">
                A research dashboard for intelligent EV charging decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = available_models()
    result_files = available_results()

    dataset_size = (
        f"{DATA_FILE.stat().st_size / (1024 * 1024):.1f} MB"
        if DATA_FILE.exists()
        else "Not found"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Training dataset", dataset_size, "Processed charging data")

    with c2:
        metric_card("Saved Q-tables", str(len(models)), "Available Q-learning models")

    with c3:
        metric_card("Evaluation files", str(len(result_files)), "Saved experiment CSVs")

    st.markdown('<div class="section-title">Your workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Train, evaluate and inspect your charging policies.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        info_card(
            "01",
            "Train the agent",
            "Configure and train a Q-learning model using the processed EV dataset.",
        )
        if st.button("Open training workspace →", key="go_train", use_container_width=True):
            st.session_state["requested_page"] = "Train model"
            st.rerun()

    with col2:
        info_card(
            "02",
            "Compare charging policies",
            "Compare immediate, price-based, Q-learning and DQN charging strategies.",
        )
        if st.button("Explore strategy comparison →", key="go_compare", use_container_width=True):
            st.session_state["requested_page"] = "Strategy Comparison"
            st.rerun()

    col3, col4 = st.columns(2, gap="large")

    with col3:
        info_card(
            "03",
            "Evaluate a model",
            "Run the existing evaluation pipeline for a saved Q-learning model.",
        )
        if st.button("Open evaluation workspace →", key="go_evaluate", use_container_width=True):
            st.session_state["requested_page"] = "Evaluate agent"
            st.rerun()

    with col4:
        info_card(
            "04",
            "Inspect results",
            "Review experiment CSVs and download data for your project report.",
        )
        if st.button("View results →", key="go_results", use_container_width=True):
            st.session_state["requested_page"] = "Results"
            st.rerun()

    st.markdown('<div class="section-title">Project health</div>', unsafe_allow_html=True)

    health1, health2 = st.columns(2)

    with health1:
        if DATA_FILE.exists():
            st.success(f"Dataset found: `{DATA_FILE.relative_to(ROOT)}`", icon="✅")
        else:
            st.error("Processed dataset not found.", icon="❌")

    with health2:
        if DEFAULT_MODEL.exists():
            st.success("Default Q-learning model found.", icon="✅")
        elif models:
            st.info("Saved models are available for evaluation.", icon="ℹ️")
        else:
            st.warning("No Q-learning model files found yet.", icon="⚠️")


# ============================================================
# TRAINING PAGE
# ============================================================

elif selected_page == "Train model":

    show_page_heading(
        "MODEL DEVELOPMENT",
        "Train your Q-learning agent",
        "Configure the learning process and generate a new Q-table.",
    )

    if not DATA_FILE.exists():
        st.error("Missing dataset: `data/processed/ev_charging_processed.csv`")
        st.stop()

    with st.form("training_form"):

        st.markdown("### Training configuration")

        col1, col2 = st.columns(2)

        with col1:
            episodes = st.number_input(
                "Training episodes",
                min_value=1,
                max_value=100000,
                value=10,
                step=10,
            )

            episode_len = st.number_input(
                "Steps per episode",
                min_value=1,
                max_value=35040,
                value=96,
                step=24,
            )

            alpha = st.slider(
                "Learning rate (alpha)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
            )

        with col2:
            gamma = st.slider(
                "Discount factor (gamma)",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
            )

            epsilon_end = st.slider(
                "Final exploration rate",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
            )

            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=1000000,
                value=42,
                step=1,
            )

        output_name = st.text_input(
            "Output model filename",
            value="q_learning_dashboard.npz",
        )

        submitted = st.form_submit_button(
            "Start training",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        output_name = output_name.strip()

        if (
            not output_name
            or Path(output_name).name != output_name
            or not output_name.endswith(".npz")
        ):
            st.error("Enter a filename ending in `.npz` without a directory path.")

        elif (MODEL_DIR / output_name).exists():
            st.error("That model filename already exists. Choose another name.")

        else:
            command = [
                sys.executable,
                "-m",
                "training.train_q_learning",
                "--episodes", str(int(episodes)),
                "--episode-len", str(int(episode_len)),
                "--alpha", str(alpha),
                "--gamma", str(gamma),
                "--epsilon-end", str(epsilon_end),
                "--seed", str(int(seed)),
                "--output", str(MODEL_DIR / output_name),
            ]

            with st.spinner("Training Q-learning agent..."):
                return_code, stdout, stderr = run_command(command)

            show_command_result(return_code, stdout, stderr)

            output_path = MODEL_DIR / output_name

            if return_code == 0 and output_path.exists():
                st.session_state["last_trained_model"] = str(output_path)
                st.success(f"Model saved to `{output_path.relative_to(ROOT)}`")


# ============================================================
# EVALUATION PAGE
# ============================================================

elif selected_page == "Evaluate agent":

    show_page_heading(
        "POLICY EVALUATION",
        "Evaluate your agent",
        "Run the existing Q-learning evaluation pipeline.",
    )

    models = available_models()

    if not models:
        st.warning("No Q-learning models found. Train an agent first.")
        if st.button("Go to training →", type="primary"):
            st.session_state["requested_page"] = "Train model"
            st.rerun()
        st.stop()

    model_options = [str(path) for path in models]
    last_model = st.session_state.get("last_trained_model")

    if last_model in model_options:
        default_index = model_options.index(last_model)
    elif str(DEFAULT_MODEL) in model_options:
        default_index = model_options.index(str(DEFAULT_MODEL))
    else:
        default_index = 0

    with st.form("evaluation_form"):

        selected_model = st.selectbox(
            "Choose a trained Q-learning model",
            options=model_options,
            index=default_index,
            format_func=lambda path: Path(path).name,
        )

        col1, col2 = st.columns(2)

        with col1:
            eval_episodes = st.number_input(
                "Evaluation episodes",
                min_value=1,
                max_value=10000,
                value=5,
                step=1,
            )

        with col2:
            eval_length = st.number_input(
                "Steps per episode",
                min_value=1,
                max_value=35040,
                value=96,
                step=24,
            )

        output_name = st.text_input(
            "Output CSV filename",
            value="dashboard_evaluation.csv",
        )

        evaluate_button = st.form_submit_button(
            "Run evaluation",
            type="primary",
            use_container_width=True,
        )

    if evaluate_button:
        output_name = output_name.strip()

        if (
            not output_name
            or Path(output_name).name != output_name
            or not output_name.endswith(".csv")
        ):
            st.error("Enter a CSV filename without a directory path.")

        else:
            output_path = RESULTS_DIR / output_name

            command = [
                sys.executable,
                "-m",
                "evaluation.evaluate",
                "--model", selected_model,
                "--episodes", str(int(eval_episodes)),
                "--episode-len", str(int(eval_length)),
                "--output", str(output_path),
            ]

            with st.spinner("Evaluating the selected model..."):
                return_code, stdout, stderr = run_command(command)

            show_command_result(return_code, stdout, stderr)

            if return_code == 0 and output_path.exists():
                df = load_results(output_path)
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                download_csv_button(
                    output_path,
                    "Download evaluation CSV",
                    "download_single_evaluation",
                )


# ============================================================
# STRATEGY COMPARISON PAGE
# ============================================================

elif selected_page == "Strategy Comparison":

    show_page_heading(
        "EXPERIMENT ANALYSIS",
        "Strategy comparison",
        "Compare charging policies using measured evaluation results.",
    )

    if not SUMMARY_FILE.exists():
        st.warning("Comparison summary is missing. Run your strategy comparison script first.")
        st.stop()

    summary = load_results(SUMMARY_FILE)

    if summary.empty or "strategy" not in summary.columns:
        st.error("The comparison summary could not be read.")
        st.stop()

    metric_columns = {
        "total_charging_cost": "Total Charging Cost",
        "total_energy_kwh": "Total Energy Consumed (kWh)",
        "peak_grid_load_kw": "Peak Grid Load (kW)",
        "final_mean_soc_percent": "Final Mean SOC (%)",
        "average_charging_delay_minutes": "Average Charging Delay (min)",
        "deadline_violations": "Deadline Violations",
        "charging_success_rate_percent": "Charging Success Rate (%)",
    }

    st.markdown("### Performance summary")

    available_metric_columns = [
        column for column in metric_columns if column in summary.columns
    ]

    display_summary = summary[["strategy"] + available_metric_columns].rename(
        columns={
            "strategy": "Charging Strategy",
            **metric_columns,
        }
    )

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These values are averages across the evaluated windows. "
        "Lower cost, energy, grid load, delay and violations are generally "
        "preferable; a higher charging success rate is better."
    )

    # Cost comparison
    st.markdown("### 1. Charging cost comparison")

    if "total_charging_cost" in summary.columns:
        st.bar_chart(
            summary.set_index("strategy")[["total_charging_cost"]]
        )

    # Energy comparison
    st.markdown("### 2. Energy consumption comparison")

    if "total_energy_kwh" in summary.columns:
        st.bar_chart(
            summary.set_index("strategy")[["total_energy_kwh"]]
        )

    # SOC and grid load
    if TIMESERIES_FILE.exists():
        timeseries = load_results(TIMESERIES_FILE)

        if not timeseries.empty and "strategy" in timeseries.columns:

            strategy_options = timeseries["strategy"].dropna().unique().tolist()

            selected_strategy = st.selectbox(
                "Choose strategy for time-series analysis",
                strategy_options,
                key="comparison_strategy",
            )

            filtered = timeseries[
                timeseries["strategy"] == selected_strategy
            ].copy()

            if "window_start" in filtered.columns:
                windows = sorted(filtered["window_start"].dropna().unique())

                if windows:
                    selected_window = st.selectbox(
                        "Evaluation window",
                        windows,
                        key="comparison_window",
                    )

                    filtered = filtered[
                        filtered["window_start"] == selected_window
                    ]

            if "time_hours" in filtered.columns:
                filtered = filtered.sort_values("time_hours")

                if "mean_connected_soc" in filtered.columns:
                    st.markdown("### 3. Mean state of charge over time")

                    soc_data = filtered.set_index("time_hours")[
                        ["mean_connected_soc"]
                    ].rename(
                        columns={
                            "mean_connected_soc": "Mean Connected SOC"
                        }
                    )

                    st.line_chart(soc_data)

                if "grid_load_kw" in filtered.columns:
                    st.markdown("### 4. Grid load over time")
                    grid_data = filtered.set_index("time_hours")[["grid_load_kw"]]
                    st.line_chart(grid_data)

    else:
        st.warning("Comparison time-series data is missing.")

    # DQN training graph
    st.markdown("### 5. DQN training performance")

    if DQN_REWARDS_FILE.exists():
        rewards = load_results(DQN_REWARDS_FILE)

        if {"episode", "total_reward"}.issubset(rewards.columns):
            reward_data = rewards.set_index("episode")[["total_reward"]].rename(
                columns={"total_reward": "Episode Reward"}
            )

            st.line_chart(reward_data)

            st.caption(
                "This training run used five episodes as an initial smoke test. "
                "The reward curve is preliminary and does not demonstrate convergence."
            )

            download_csv_button(
                DQN_REWARDS_FILE,
                "Download DQN training history",
                "download_dqn_rewards",
            )
        else:
            st.warning("The DQN training history has unexpected columns.")
    else:
        st.info("DQN training history has not been generated yet.")

    st.markdown("### Download comparison data")

    download_csv_button(
        SUMMARY_FILE,
        "Download strategy summary",
        "download_summary",
    )

    download_csv_button(
        DETAILS_FILE,
        "Download detailed evaluation",
        "download_details",
    )

    download_csv_button(
        TIMESERIES_FILE,
        "Download time-series data",
        "download_timeseries",
    )

    if PLOTS_DIR.exists():
        st.markdown("### Saved graphs")

        graph_files = [
            ("cost_comparison.png", "Charging cost comparison"),
            ("energy_comparison.png", "Energy consumption comparison"),
            ("soc_over_time.png", "SOC over time"),
            ("grid_load_over_time.png", "Grid load over time"),
            ("rl_training_performance.png", "DQN training performance"),
        ]

        for filename, caption in graph_files:
            graph_path = PLOTS_DIR / filename

            if graph_path.exists():
                with st.expander(caption):
                    st.image(str(graph_path), caption=caption, use_container_width=True)


# ============================================================
# RESULTS PAGE
# ============================================================

elif selected_page == "Results":

    show_page_heading(
        "EXPERIMENT DATA",
        "Results & downloads",
        "Inspect saved CSV files and export experiment results.",
    )

    result_files = available_results()

    if not result_files:
        st.info("No CSV results are available yet.")
        st.stop()

    # Prefer the strategy summary as the default file.
    result_options = [str(path) for path in result_files]

    if str(SUMMARY_FILE) in result_options:
        default_result_index = result_options.index(str(SUMMARY_FILE))
    else:
        default_result_index = 0

    selected_result = st.selectbox(
        "Choose an evaluation CSV",
        options=result_options,
        index=default_result_index,
        format_func=lambda path: Path(path).name,
    )

    result_path = Path(selected_result)
    df = load_results(result_path)

    if df.empty:
        st.warning("The selected CSV is empty or could not be read.")
        st.stop()

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:
        st.markdown("### Explore a metric")

        selected_metric = st.selectbox(
            "Numeric metric",
            numeric_columns,
            key="results_metric",
        )

        possible_labels = [
            column
            for column in df.columns
            if column not in numeric_columns
            and df[column].nunique(dropna=True) <= 30
        ]

        if possible_labels:
            label_column = st.selectbox(
                "Group by",
                ["None"] + possible_labels,
                key="results_group",
            )

            if label_column != "None":
                chart_df = df[[label_column, selected_metric]].copy()
                chart_df = chart_df.set_index(label_column)
                st.bar_chart(chart_df)
            else:
                st.line_chart(df[[selected_metric]])
        else:
            st.line_chart(df[[selected_metric]])

    download_csv_button(
        result_path,
        "Download selected CSV",
        "download_selected_csv",
    )

    st.caption(
        "These charts represent saved experimental data, not live charging-station telemetry."
    )


# ============================================================
# FOOTER
# ============================================================

st.write("")
st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#758078;font-size:0.8rem;padding:10px 0 20px 0;">
        EVora · Smart Charging Intelligence<br>
        Reinforcement learning research dashboard
    </div>
    """,
    unsafe_allow_html=True,
)