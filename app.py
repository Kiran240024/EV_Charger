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

    .hero-symbol {
        font-size: 5rem;
        line-height: 1;
        margin-bottom: 12px;
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

    .status-pill {
        display: inline-block;
        border-radius: 100px;
        padding: 6px 12px;
        background: #E6F2EA;
        color: #216647;
        font-size: 0.76rem;
        font-weight: 700;
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

    #MainMenu {
        visibility: hidden;
    }

    footer {
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
    """Execute a project command and return its output."""
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

    models = available_models()

    st.caption(f"Saved models · {len(models)}")
    st.caption(f"Evaluation files · {len(available_results())}")

    st.divider()

    st.markdown(
        """
        <div class="small-muted">
            EVora · Reinforcement learning<br>
            Q-learning based EV charging
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
                Explore how a Q-learning agent makes electric vehicle
                charging decisions while balancing energy costs,
                grid demand and battery requirements.
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
        metric_card(
            "Training dataset",
            dataset_size,
            "Processed charging data",
        )

    with c2:
        metric_card(
            "Saved Q-tables",
            str(len(models)),
            "Available model files",
        )

    with c3:
        metric_card(
            "Evaluation runs",
            str(len(result_files)),
            "Saved CSV result files",
        )

    st.markdown(
        '<div class="section-title">Your workspace</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Train, evaluate and inspect your charging agent.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        info_card(
            "01",
            "Train the agent",
            "Run Q-learning on the processed EV charging dataset. "
            "Choose the number of episodes and learning parameters, "
            "then save a new model.",
        )

        if st.button(
            "Open training workspace →",
            key="go_train",
            use_container_width=True,
        ):
            st.session_state["requested_page"] = "Train model"
            st.rerun()

    with col2:
        info_card(
            "02",
            "Evaluate performance",
            "Compare the learned policy with baseline charging "
            "strategies and inspect the metrics generated by your "
            "evaluation script.",
        )

        if st.button(
            "Open evaluation workspace →",
            key="go_evaluate",
            use_container_width=True,
        ):
            st.session_state["requested_page"] = "Evaluate agent"
            st.rerun()

    st.write("")

    col3, col4 = st.columns(2, gap="large")

    with col3:
        info_card(
            "03",
            "Explore results",
            "Review evaluation CSV files, inspect numeric metrics "
            "and download the results for your project report.",
        )

        if st.button(
            "View evaluation results →",
            key="go_results",
            use_container_width=True,
        ):
            st.session_state["requested_page"] = "Results"
            st.rerun()

    with col4:
        info_card(
            "04",
            "Understand the workflow",
            "The environment models EV charging decisions. "
            "The agent learns a policy, and the evaluation module "
            "compares that policy against baseline strategies.",
        )

    st.write("")

    st.markdown(
        '<div class="section-title">Project health</div>',
        unsafe_allow_html=True,
    )

    health1, health2 = st.columns(2)

    with health1:
        if DATA_FILE.exists():
            st.success(
                f"Dataset found: `{DATA_FILE.relative_to(ROOT)}`",
                icon="✅",
            )
        else:
            st.error(
                "The processed dataset was not found at "
                "`data/processed/ev_charging_processed.csv`.",
                icon="❌",
            )

    with health2:
        if DEFAULT_MODEL.exists():
            st.success(
                "Default 200-episode model found.",
                icon="✅",
            )
        elif models:
            st.info(
                "Saved models are available. Select one on the evaluation page.",
                icon="ℹ️",
            )
        else:
            st.warning(
                "No saved models found. Train a model to get started.",
                icon="⚠️",
            )


# ============================================================
# TRAINING PAGE
# ============================================================

elif selected_page == "Train model":

    show_page_heading(
        "MODEL DEVELOPMENT",
        "Train your agent",
        "Configure the learning process and generate a new Q-learning model.",
    )

    if not DATA_FILE.exists():
        st.error(
            "The training dataset is missing. Expected location: "
            "`data/processed/ev_charging_processed.csv`"
        )
        st.stop()

    st.markdown(
        '<div class="section-title">Training configuration</div>',
        unsafe_allow_html=True,
    )

    with st.form("training_form"):

        col1, col2 = st.columns(2)

        with col1:
            episodes = st.number_input(
                "Training episodes",
                min_value=1,
                max_value=100000,
                value=10,
                step=10,
                help="Use a small value for a quick test.",
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
            help="The file is saved inside the models directory.",
        )

        st.caption(
            "Tip: Start with 10 episodes to verify the workflow. "
            "Increase the number of episodes for more extensive experiments."
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
            st.error(
                "Enter a filename ending in `.npz`, "
                "without a directory path."
            )

        elif (MODEL_DIR / output_name).exists():
            st.error(
                f"`{output_name}` already exists. "
                "Choose another filename to avoid overwriting it."
            )

        else:
            command = [
                sys.executable,
                "-m",
                "training.train_q_learning",
                "--episodes",
                str(int(episodes)),
                "--episode-len",
                str(int(episode_len)),
                "--alpha",
                str(alpha),
                "--gamma",
                str(gamma),
                "--epsilon-end",
                str(epsilon_end),
                "--seed",
                str(int(seed)),
                "--output",
                str(MODEL_DIR / output_name),
            ]

            st.markdown("#### Training progress")

            with st.spinner(
                "Training is running. This may take a little while..."
            ):
                return_code, stdout, stderr = run_command(command)

            show_command_result(return_code, stdout, stderr)

            if return_code == 0:
                output_path = MODEL_DIR / output_name

                if output_path.exists():
                    st.session_state["last_trained_model"] = str(output_path)

                    st.success(
                        f"Model saved to `{output_path.relative_to(ROOT)}`"
                    )

                    st.balloons()


# ============================================================
# EVALUATION PAGE
# ============================================================

elif selected_page == "Evaluate agent":

    show_page_heading(
        "POLICY EVALUATION",
        "Put the agent to the test",
        "Evaluate a saved Q-learning policy against baseline charging strategies.",
    )

    models = available_models()

    if not models:
        st.warning(
            "No model files were found. Train an agent before evaluating it."
        )

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
            "Choose a trained model",
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
            st.error(
                "Enter a filename ending in `.csv`, "
                "without a directory path."
            )

        else:
            output_path = RESULTS_DIR / output_name

            command = [
                sys.executable,
                "-m",
                "evaluation.evaluate",
                "--model",
                selected_model,
                "--episodes",
                str(int(eval_episodes)),
                "--episode-len",
                str(int(eval_length)),
                "--output",
                str(output_path),
            ]

            with st.spinner(
                "Evaluating the selected model and baseline policies..."
            ):
                return_code, stdout, stderr = run_command(command)

            show_command_result(return_code, stdout, stderr)

            if return_code == 0 and output_path.exists():
                st.success(
                    f"Evaluation saved to `{output_path.relative_to(ROOT)}`"
                )

                df = load_results(output_path)

                if not df.empty:
                    st.markdown("#### Evaluation output")
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        "Download evaluation CSV",
                        data=output_path.read_bytes(),
                        file_name=output_path.name,
                        mime="text/csv",
                    )
                else:
                    st.warning(
                        "The output file exists, but no readable rows "
                        "were found in it."
                    )


# ============================================================
# RESULTS PAGE
# ============================================================

elif selected_page == "Results":

    show_page_heading(
        "EXPERIMENT ANALYSIS",
        "Results & insights",
        "Inspect saved evaluation results and explore how the charging policies perform.",
    )

    result_files = available_results()

    if not result_files:
        st.info(
            "No evaluation CSV files found yet. Run an evaluation to populate this page."
        )

        if st.button("Evaluate a model →", type="primary"):
            st.session_state["requested_page"] = "Evaluate agent"
            st.rerun()

        st.stop()

    selected_result = st.selectbox(
        "Choose an evaluation file",
        options=[str(path) for path in result_files],
        format_func=lambda path: Path(path).name,
    )

    df = load_results(Path(selected_result))

    if df.empty:
        st.warning(
            "This CSV could not be loaded or contains no rows."
        )
        st.stop()

    st.markdown("#### Evaluation data")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:

        st.markdown("#### Explore a metric")

        selected_metric = st.selectbox(
            "Numeric metric",
            numeric_columns,
        )

        chart_data = df[[selected_metric]].copy()

        if len(df.columns) > 1:
            possible_labels = [
                column
                for column in df.columns
                if column not in numeric_columns
                and df[column].nunique(dropna=True) <= 30
            ]

            if possible_labels:
                label_column = st.selectbox(
                    "Group by (optional)",
                    ["None"] + possible_labels,
                )

                if label_column != "None":
                    chart_data = df[[label_column, selected_metric]].copy()
                    chart_data = chart_data.set_index(label_column)

        st.line_chart(chart_data)

    else:
        st.info(
            "This CSV does not contain numeric columns to plot."
        )

    st.download_button(
        "Download this CSV",
        data=Path(selected_result).read_bytes(),
        file_name=Path(selected_result).name,
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "These charts display saved experiment data. "
        "They do not represent live charging-station telemetry."
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
        Q-learning research dashboard
    </div>
    """,
    unsafe_allow_html=True,
)