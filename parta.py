import numpy as np
from create_stochastic import P, reachable_states

gamma = 0.9
reward_on_entry = 10.0

# Identify terminal states
def is_terminal(state):
    return state[0] == 1 or state[1] == 1

terminal_flags = np.array([is_terminal(s) for s in reachable_states])
n = len(reachable_states)

# Compute transition-based rewards (same as before)
r_trans = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if terminal_flags[j] and not terminal_flags[i]:
            r_trans[i, j] = reward_on_entry
        else:
            r_trans[i, j] = -.1

# Simulate returns via rollouts
def compute_return_from_state(start_state, steps=10):
    """Monte Carlo estimate of return from a given state."""
    i = start_state
    total_return = 0.0
    discount = 0.9

    for _ in range(steps):
        # Sample next state based on transition probabilities
        j = np.random.choice(range(n), p=P[i])
        total_return += discount * r_trans[i, j]
        if terminal_flags[j]:
            break  # stop once a terminal state is reached
        discount *= gamma
        i = j

    return total_return

# Compute expected return for all states
returns = np.array([np.mean([compute_return_from_state(i) for _ in range(1000)]) for i in range(n)])

print("Expected Monte Carlo Returns per State:\n", np.round(returns, 3))