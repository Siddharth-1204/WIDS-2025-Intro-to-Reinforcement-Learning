import matplotlib.pyplot as plt

# ---------------- PROBLEM PARAMETERS ----------------

Goal = 100            # target capital
p_h = 0.4             # probability of winning a bet (heads)
theta = 1e-9          # convergence threshold for value iteration

# ---------------- INITIALIZATION ----------------

# Value function: V[s] = probability of eventually reaching 100 from capital s
V = [0.0] * (Goal + 1)
V[Goal] = 1.0         # terminal state has value 1

# Policy: policy[s] = optimal stake at capital s
policy = [0] * (Goal + 1)

# ---------------- VALUE ITERATION ----------------

while True:
    delta = 0
    for s in range(1, Goal):
        old_v = V[s]
        action_values = []

        # Possible stakes: 1 to min(s, 100 - s)
        for a in range(1, min(s, Goal - s) + 1):
            win_state = s + a
            lose_state = s - a

            # Reward only when reaching the goal
            reward = 1 if win_state == Goal else 0

            # Bellman optimality update
            val = (
                p_h * (reward + V[win_state]) +
                (1 - p_h) * V[lose_state]
            )
            action_values.append(val)

        # Update value function
        V[s] = max(action_values)
        delta = max(delta, abs(old_v - V[s]))

    # Stop when values have converged
    if delta < theta:
        break

# ---------------- POLICY EXTRACTION ----------------

for s in range(1, Goal):
    action_values = {}

    # Compute action-value for each possible stake
    for a in range(1, min(s, Goal - s) + 1):
        win_state = s + a
        lose_state = s - a
        reward = 1 if win_state == Goal else 0

        action_values[a] = (
            p_h * (reward + V[win_state]) +
            (1 - p_h) * V[lose_state]
        )

    # Find actions achieving the maximum value
    max_value = max(action_values.values())
    best_actions = [
        a for a, v in action_values.items()
        if abs(v - max_value) < 1e-12
    ]

    # Choose the smallest optimal stake (book-style tie-breaking)
    policy[s] = min(best_actions)

# ---------------- PLOT OPTIMAL POLICY ----------------

plt.figure(figsize=(10, 5))
plt.bar(range(Goal + 1), policy)
plt.xlabel("Capital")
plt.ylabel("Stake (Action)")
plt.title("Optimal Policy for Gambler's Problem")
plt.show()