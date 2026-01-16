# ---------------- LIGHTS OUT (4x4) : PART 3 ----------------

N = 4
NUM_CELLS = 16

TERMINAL_STATE = "0b" + "1" * NUM_CELLS

# ---------------- BUILD STATE SPACE ----------------

state_space = []
for i in range(2**NUM_CELLS):                # 0–65535
    bitstring = bin(i)[2:].zfill(NUM_CELLS)  # exactly 16 bits
    state_space.append("0b" + bitstring)

# ---------------- TOGGLE FUNCTION ----------------

def toggle(state, indices):
    bits = list(state[2:])   # extract 16 bits

    for cell in indices:
        idx = cell - 1       # convert 1–16 → 0–15
        bits[idx] = "0" if bits[idx] == "1" else "1"

    return "0b" + "".join(bits)

# ---------------- NEIGHBORS ----------------

def neighbors(cell):
    row = (cell - 1) // N
    col = (cell - 1) % N
    nbrs = [cell]

    if row > 0:       nbrs.append(cell - N)
    if row < N - 1:   nbrs.append(cell + N)
    if col > 0:       nbrs.append(cell - 1)
    if col < N - 1:   nbrs.append(cell + 1)

    return nbrs

# ---------------- TRANSITION FUNCTION ----------------

def transition_dict(state):
    d = {}

    for action in range(1, NUM_CELLS + 1):

        if state == TERMINAL_STATE:
            d[action] = [(1, state, 0, True)]
            continue

        affected = neighbors(action)
        next_state = toggle(state, affected)
        done = (next_state == TERMINAL_STATE)

        d[action] = [(1, next_state, int(done), done)]

    return d

# ---------------- BUILD MDP ----------------

lo_map = {state: transition_dict(state) for state in state_space}

# ---------------- VALUE ITERATION ----------------

V = {state: (0 if state == TERMINAL_STATE else float("inf")) for state in state_space}

changed = True
while changed:
    changed = False

    for state in state_space:
        if state == TERMINAL_STATE:
            continue

        best = float("inf")
        for action in lo_map[state]:
            _, next_state, _, _ = lo_map[state][action][0]
            best = min(best, 1 + V[next_state])

        if best < V[state]:
            V[state] = best
            changed = True

# ---------------- EXTRACT POLICY ----------------

policy = {}

for state in state_space:
    if state == TERMINAL_STATE:
        continue

    best_action = None
    best_val = float("inf")

    for action in lo_map[state]:
        _, next_state, _, _ = lo_map[state][action][0]
        val = 1 + V[next_state]
        if val < best_val:
            best_val = val
            best_action = action

    policy[state] = best_action

# ---------------- FINAL ANSWER ----------------

min_moves_guarantee = max(V.values())

print("Minimum number of moves to guarantee solving any board: ", min_moves_guarantee)
