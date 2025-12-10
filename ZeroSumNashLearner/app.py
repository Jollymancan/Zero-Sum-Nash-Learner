import numpy as np
import streamlit as st
from core import learn_nash_zero_sum

st.set_page_config(page_title="Zero-Sum Game Nash Learner", layout="wide")

st.title("Zero-Sum Game Nash Learner")

st.markdown(
    """
    This app learns an approximate Nash equilibrium for a 2-player **zero-sum** normal-form game.

    - You specify the payoff matrix **M** for Player 1 (rows).
    - Player 2's payoff is **-M**.
    - We run a multiplicative-weights style learning dynamic that converges to a Nash equilibrium in zero-sum games.
    """
)

# --- Sidebar: presets & hyperparams ---
st.sidebar.header("Game & Hyperparameters")

preset = st.sidebar.selectbox(
    "Preset game",
    ["Custom", "Rock–Paper–Scissors", "Matching Pennies"]
)

default_matrix = "0 -1 1\n1 0 -1\n-1 1 0"  # RPS

if preset == "Matching Pennies":
    default_matrix = "1 -1\n-1 1"
elif preset == "Custom":
    default_matrix = "0 -1 1\n1 0 -1\n-1 1 0"

matrix_text = st.text_area(
    "Payoff matrix M for Player 1 (rows: space-separated, one row per line):",
    value=default_matrix,
    height=150,
)

lr = st.sidebar.number_input("Learning rate (lr)", min_value=1e-4, max_value=1.0, value=0.1, step=0.01, format="%.4f")
T = st.sidebar.number_input("Iterations (T)", min_value=1000, max_value=500000, value=100000, step=1000)
log_interval = st.sidebar.number_input("Log interval", min_value=100, max_value=50000, value=5000, step=100)

run_button = st.button("Run Nash Learner")

def parse_matrix(text):
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        row = [float(x) for x in line.strip().split()]
        rows.append(row)
    if not rows:
        raise ValueError("No rows parsed.")
    n_cols = len(rows[0])
    for r in rows:
        if len(r) != n_cols:
            raise ValueError("All rows must have the same number of columns.")
    return np.array(rows, dtype=float)

if run_button:
    try:
        M = parse_matrix(matrix_text)
        n1, n2 = M.shape
        st.write(f"Matrix shape: **{n1} × {n2}**")

        with st.spinner("Learning approximate Nash equilibrium..."):
            p, q, history = learn_nash_zero_sum(
                M,
                lr=float(lr),
                T=int(T),
                log_interval=int(log_interval),
                seed=0,
            )

        st.subheader("Approximate Nash Equilibrium")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Player 1 (row) strategy p:**")
            for i, prob in enumerate(p):
                st.write(f"Action {i}: {prob:.4f}")
        with col2:
            st.write("**Player 2 (column) strategy q:**")
            for j, prob in enumerate(q):
                st.write(f"Action {j}: {prob:.4f}")

        # Current game value
        game_value = float(p @ M @ q)
        st.markdown(f"**Estimated game value for Player 1:** {game_value:.4f}")

        # Turn history into arrays for plotting
        if history:
            iters = np.array([h[0] for h in history])
            values = np.array([h[1] for h in history])
            expl_p1 = np.array([h[2] for h in history])
            expl_p2 = np.array([h[3] for h in history])
            expl_max = np.array([h[4] for h in history])

            st.subheader("Training curves")

            tab1, tab2 = st.tabs(["Exploitability", "Game value"])

            with tab1:
                st.line_chart(
                    {
                        "iter": iters,
                        "P1 exploitability": expl_p1,
                        "P2 exploitability": expl_p2,
                        "max exploitability": expl_max,
                    },
                    x="iter",
                )

            with tab2:
                st.line_chart(
                    {
                        "iter": iters,
                        "game value (P1)": values,
                    },
                    x="iter",
                )

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
