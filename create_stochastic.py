import numpy as np

# Capacities
CAP_A = 3
CAP_B = 2

# List only reachable states from (0,0)
reachable_states = [
    (0,0), (0,1), (0,2),
    (1,0), (1,2),
    (2,0), (2,2),
    (3,0), (3,1), (3,2)
]

state_index = {state: i for i, state in enumerate(reachable_states)}
num_states = len(reachable_states)

# Actions
def fill_a(a, b):
    return (CAP_A, b)

def fill_b(a, b):
    return (a, CAP_B)

def empty_a(a, b):
    return (0, b)

def empty_b(a, b):
    return (a, 0)

def pour_a_to_b(a, b):
    pour = min(a, CAP_B - b)
    return (a - pour, b + pour)

def pour_b_to_a(a, b):
    pour = min(b, CAP_A - a)
    return (a + pour, b - pour)

actions = [fill_a, fill_b, empty_a, empty_b, pour_a_to_b, pour_b_to_a]

# Initialize stochastic matrix
P = np.zeros((num_states, num_states))

for state in reachable_states:
    idx = state_index[state]
    
    # Check if state has exactly 1 liter in any bucket
    if state[0] == 1 or state[1] == 1:
        # Make row “terminal-like”: only self-loop
        P[idx, idx] = 1.0
        continue
    
    for action in actions:
        next_state = action(*state)
        # Only include next_state if it's in reachable_states
        if next_state in state_index:
            next_idx = state_index[next_state]
            P[idx, next_idx] += 1 / len(actions)

# Print results
np.set_printoptions(precision=2, suppress=True)
# # Initialize reward vector
R = np.zeros(num_states)

# Set reward = 10 for terminal states (states with 1 liter in any bucket)
for state in reachable_states:
    idx = state_index[state]
    if state[0] == 1 or state[1] == 1:
        R[idx] = 10

gamma = 0.9
I = np.eye(num_states)  # Identity matrix of size 10x10

# Compute (I - gamma * P)
I_minus_gammaP = I - gamma * P

# print("I - gamma * P:\n", I_minus_gammaP)
