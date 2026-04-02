D2D_SKIP = (-1, -1)


def build_action_table(n_steps):
    """Triangular grid of (pi_r, pi_d) pairs where pi_r + pi_d <= 1, plus D2D skip."""
    table = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            table.append((round(i / n_steps, 6), round(j / n_steps, 6)))
    table.append(D2D_SKIP)
    return table


def action_to_splits(action_idx, table):
    return table[action_idx]
