import numpy as np

def learn_nash_zero_sum(M, lr=0.1, T=200_000, log_interval=20_000, seed=0):
    rng = np.random.default_rng(seed)
    M = np.asarray(M, float)
    n1, n2 = M.shape

    p = np.ones(n1) / n1
    q = np.ones(n2) / n2

    p_avg = np.zeros_like(p)
    q_avg = np.zeros_like(q)

    history = []

    for t in range(1, T + 1):
        u = M @ q
        v = M.T @ p

        u_c = u - u.mean()
        v_c = v - v.mean()

        p = p * np.exp(lr * u_c)
        p /= p.sum()

        q = q * np.exp(-lr * v_c)
        q /= q.sum()

        # running averages
        p_avg += p
        q_avg += q

        if t % log_interval == 0 or t == T:
            val = float(p @ M @ q)
            br1_val = float((M @ q).max())
            expl_p1 = br1_val - val

            col_payoffs = p @ M
            br2_val = float(-(col_payoffs.min()))
            val2 = float(-val)
            expl_p2 = br2_val - val2
            expl_max = max(expl_p1, expl_p2)

            history.append((t, val, expl_p1, expl_p2, expl_max))

    p_avg /= T
    q_avg /= T

    return p_avg, q_avg, history


if __name__ == "__main__":
    # Actions: 0 = Rock, 1 = Paper, 2 = Scissors
    M_rps = np.array([
        [ 1, -1],  # Rock vs (R, P, S)
        [ -1,  2],  # Paper
    ], dtype=float)

    p, q, hist = learn_nash_zero_sum(M_rps, lr=0.1, T=200_000, log_interval=20_000)

    print("Approximate Nash strategies:")
    print("Player 1 (rows):   ", p)
    print("Player 2 (columns):", q)
    print()

    print("Last few log entries (t, value, expl_p1, expl_p2, expl_max):")
    for row in hist[-5:]:
        print(row)
