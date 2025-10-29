from create_stochastic import P, I_minus_gammaP, reachable_states
import numpy as np

gamma = 0.9
reward_on_entry = 10.0

# Identify terminal states
def is_terminal(state):
    return state[0] == 1 or state[1] == 1

terminal_flags = [is_terminal(s) for s in reachable_states]
n = len(reachable_states)

# Transition-based reward matrix
r_trans = np.zeros((n, n))
for i, s in enumerate(reachable_states):
    for j, s_next in enumerate(reachable_states):
        # reward only when moving INTO a terminal state from a non-terminal
        if terminal_flags[j] and not terminal_flags[i]:
            r_trans[i, j] = reward_on_entry

# Compute expected immediate reward vector
r_vec = np.sum(P * r_trans, axis=1)

# Solve (I - γP)V = r_vec
V = np.linalg.solve(np.eye(n) - gamma * P, r_vec)

# Display
print("Stochastic matrix P:\n", P)
print("\nReward-on-entry matrix r_trans:\n", r_trans)
print("\nExpected immediate rewards r_vec:\n", r_vec)
print("\nValue vector V:\n", ', '.join(map(lambda x: f'{x:.3f}', V.flatten())))
