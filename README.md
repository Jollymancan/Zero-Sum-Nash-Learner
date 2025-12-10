# Zero-Sum-Nash-Learner
A small Streamlit app for experimenting with **approximate Nash equilibria** in 2-player **zero-sum** normal-form games, using a multiplicative-weights style learning dynamic.

You provide the payoff matrix **M** for Player 1 (rows); Player 2’s payoff is automatically taken as **−M**. The app then runs the learning procedure and visualizes how the game value and exploitability evolve over time.

---

## Features

- Enter any finite 2-player zero-sum game as a payoff matrix for Player 1.
- Built-in examples:
  - Rock Paper Scissors  
  - Matching Pennies
- Multiplicative-weights updates that converge (in the limit) to a Nash equilibrium in zero-sum games.
- Displays:
  - Approximate mixed strategies for both players
  - Estimated game value for Player 1
  - Training curves for exploitability and value

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
