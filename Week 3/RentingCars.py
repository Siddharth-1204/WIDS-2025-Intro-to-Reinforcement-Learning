import numpy as np
import matplotlib.pyplot as plt
from math import exp, factorial

# ---------------- PARAMETERS ----------------

Maximum_cars = 20
Max_action = 5
Rent_reward = 10
Move_cost = 2
Gamma = 0.9

lambda_req1 = 3
lambda_req2 = 4
lambda_ret1 = 3
lambda_ret2 = 2

Max_rent = 11

# ---------------- POISSON CACHE ----------------

def poisson_prob(n, lam):
    return exp(-lam) * lam**n / factorial(n)

poisson_cache = {}
for lam in [lambda_req1, lambda_req2, lambda_ret1, lambda_ret2]:
    poisson_cache[lam] = [poisson_prob(n, lam) for n in range(Max_rent + 1)]

# ---------------- STATE & POLICY ----------------

V = {(i, j): 0.0 for i in range(Maximum_cars + 1) for j in range(Maximum_cars + 1)}
policy = {(i, j): 0 for i in range(Maximum_cars + 1) for j in range(Maximum_cars + 1)}

# ---------------- EXPECTED RETURN ----------------
# action > 0 : move cars from location 1 -> location 2
# action < 0 : move cars from location 2 -> location 1

def expected_return(state, action, V):
    i, j = state

    if action > 0 and i < action:
        return -np.inf
    if action < 0 and j < -action:
        return -np.inf

    i_after = min(i - action, Maximum_cars)
    j_after = min(j + action, Maximum_cars)

    value = -Move_cost * abs(action)

    for r1 in range(Max_rent + 1):
        for r2 in range(Max_rent + 1):
            prob_rent = (
                poisson_cache[lambda_req1][r1] *
                poisson_cache[lambda_req2][r2]
            )

            real_r1 = min(i_after, r1)
            real_r2 = min(j_after, r2)

            reward = (real_r1 + real_r2) * Rent_reward

            i_left = i_after - real_r1
            j_left = j_after - real_r2

            for ret1 in range(Max_rent + 1):
                for ret2 in range(Max_rent + 1):
                    prob_ret = (
                        poisson_cache[lambda_ret1][ret1] *
                        poisson_cache[lambda_ret2][ret2]
                    )
                    prob = prob_rent * prob_ret

                    next_i = min(i_left + ret1, Maximum_cars)
                    next_j = min(j_left + ret2, Maximum_cars)

                    value += prob * (reward + Gamma * V[(next_i, next_j)])

    return value

# ---------------- POLICY EVALUATION ----------------

def policy_evaluation(V, policy, theta=1e-4):
    while True:
        delta = 0
        for i in range(Maximum_cars + 1):
            for j in range(Maximum_cars + 1):
                s = (i, j)
                old_v = V[s]
                V[s] = expected_return(s, policy[s], V)
                delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break
    return V

# ---------------- POLICY IMPROVEMENT ----------------

def policy_improvement(V, policy):
    stable = True

    for i in range(Maximum_cars + 1):
        for j in range(Maximum_cars + 1):

            s = (i, j)
            old_action = policy[s]

            action_values = {}

            for a in range(-Max_action, Max_action + 1):
                if 0 <= i - a <= Maximum_cars and 0 <= j + a <= Maximum_cars:
                    action_values[a] = expected_return(s, a, V)

            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action

            if best_action != old_action:
                stable = False

    return policy, stable

# ---------------- POLICY ITERATION ----------------

def policy_iteration():
    global V, policy
    while True:
        V = policy_evaluation(V, policy)
        policy, stable = policy_improvement(V, policy)
        if stable:
            break
    return V, policy

V_opt, policy_opt = policy_iteration()

# ---------------- HEATMAP ----------------

policy_array = np.zeros((Maximum_cars + 1, Maximum_cars + 1))
for (i, j), action in policy_opt.items():
    policy_array[i, j] = action

plt.figure(figsize=(8, 6))
plt.imshow(policy_array, origin="lower", cmap="viridis")
plt.colorbar(label="Cars moved")
plt.xlabel("Cars at location 1")
plt.ylabel("Cars at location 2")
plt.title("Optimal Policy Heatmap (Jack's Car Rental)")
plt.show()