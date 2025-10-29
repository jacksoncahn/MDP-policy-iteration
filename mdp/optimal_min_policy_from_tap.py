import numpy as np


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
def fill_a(a, b, mode='normal'):
    if mode == 'spill':
        # example: spillage returns some predefined state (or same state)
        return (a, b)  # adjust as needed
    return (CAP_A, b)

def fill_b(a, b, mode='normal'):
    if mode == 'spill':
        return (a, b)  # adjust as needed
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
num_actions = len(actions)

#create Psas
Psas = np.zeros((num_states, num_actions, num_states))
for s_idx, state in enumerate(reachable_states):
    # Terminal states: self-loop for all actions
    if state[0] == 1 or state[1] == 1:
        for a_idx in range(num_actions):
            Psas[s_idx, a_idx, s_idx] = 1.0
        continue
    
    for a_idx, action in enumerate(actions):
        if action == fill_a:
            # 0.1 spillage, 0.9 normal
            next_state_spill = fill_a(*state, mode='spill')
            next_state_normal = fill_a(*state, mode='normal')

            if next_state_spill in state_index:
                Psas[s_idx, a_idx, state_index[next_state_spill]] = 0.1
            if next_state_normal in state_index:
                Psas[s_idx, a_idx, state_index[next_state_normal]] = 0.9

        elif action == fill_b:
            # 0.9 spillage, 0.1 normal
            next_state_spill = fill_b(*state, mode='spill')
            next_state_normal = fill_b(*state, mode='normal')

            if next_state_spill in state_index:
                Psas[s_idx, a_idx, state_index[next_state_spill]] = 0.9
            if next_state_normal in state_index:
                Psas[s_idx, a_idx, state_index[next_state_normal]] = 0.1

        else:
            # Deterministic outcome for other actions
            next_state = action(*state)
            if next_state in state_index:
                Psas[s_idx, a_idx, state_index[next_state]] = 1.0

# print(Psas)
# print("\n")

#Create P
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
P = np.round(P, decimals=3)

# print(P)
# print("\n")

# Identify terminal states
def is_terminal(state):
    return state[0] == 1 or state[1] == 1

terminal_flags = [is_terminal(s) for s in reachable_states]
n = len(reachable_states)

# Compute expected immediate reward vector
# r_trans[s, a, s'] = reward for taking action a in state s and ending in s'
r_trans = np.zeros((n, num_actions, n))

for s_idx, s in enumerate(reachable_states):
    for a_idx, action in enumerate(actions):
        fill_amount = 0
        if a_idx <= 1:            
            #set all rewards to -1 for fill actions (we want to minimze fills)
            #depends on fill_a or fill_b and also on current state (how much is in the bucket already)
            #apparently I have to check each state and however much water would be added by filling (whatever space is left in the bucket)
            # so reward is -1 * (amount added to bucket; if bucket is full, no penalty)
            # hopefully I can index reachable_states to get current amount in bucket

            #REWARD VECTOR LOOKS OKAY NOW, HOPEFULLY DOES WHAT I WANT
            current_amount = reachable_states[s_idx][a_idx]
            r_trans[s_idx, a_idx, :] = -1 * ( [CAP_A, CAP_B][a_idx] - current_amount )

        for s_next_idx, s_next in enumerate(reachable_states):
            # reward only when moving INTO a terminal state from a non-terminal AND neither bucket was *meaningfully* filled
            if terminal_flags[s_next_idx] and not terminal_flags[s_idx]:
                r_trans[s_idx, a_idx, s_next_idx] = 100
        
print("RTRANS \n", r_trans)

#indexable now by current state, and action r_vec[s,a]
#R_VEC is what enforces only receiving reward when transitioning into terminal states
#therefore, P can have 1s in terminal states as self-loops without affecting rewards

#V_pi[s] = R_VEC[s, policy[s]] + γ * sum(P_pi[s, s'] * V_pi[s'] for s')
R_MATRIX = np.sum(Psas * r_trans, axis=2)  # sum over s' (axis=2)
print(R_MATRIX)

# R_MATRIX = np.zeros((num_states, num_actions))
# for s_idx in range(num_states):
#     for a_idx in range(num_actions):
#         total = 0.0
#         for s_next_idx in range(num_states):
#             total += Psas[s_idx, a_idx, s_next_idx] * r_trans[s_idx, a_idx, s_next_idx]
#         R_MATRIX[s_idx, a_idx] = total

# print(R_MATRIX)


# r_vec has shape (n, num_actions)
# Initialize policy array
# Will hold the optimal action index for each state
# will iterate through P and r_vec, then edit the probabilities as dictated by the policy array
# if state is terminal, policy can be None
# if state is not terminal, policy will be an integer index into actions, so we can set the action probabilities accordingly
# of the format state_index, (action_index or None)
POLICY = {}
for s_idx in range(num_states):
    if terminal_flags[s_idx]:
        POLICY[s_idx] = None  # terminal states stay as self-loops
    else:
        POLICY[s_idx] = 2 # random initial action


#for each state in the policy P[state, *] = 0, then set P[state, action_index] = 1.0
#we need this to update transition matrix w.r.t. the policy
def p_wrt_policy(P, policy):
    P_pi = np.zeros((num_states, num_states))
    for s_idx in range(num_states):  
        if policy[s_idx] is None:
            # Terminal state: self-loop
            P_pi[s_idx, s_idx] = 1.0
        else:
            a_idx = policy[s_idx]
            P_pi[s_idx, :] = P[s_idx, a_idx, :]
    return P_pi

print(P)
print (p_wrt_policy(Psas, POLICY))

#computes V_pi for each state (we need this to evaluate policies)
#discount factor gamma is superflous for this MDP, but it can't be 1 or it will break the matrix inversion step
def value_computation(P, R,  policy, gamma=0.9):
    n = len(policy)
    R_pi = np.zeros(n)
    # Get expected reward for the action chosen by the policy in each state
    for s_idx in range(n):
        if policy[s_idx] is None:
            R_pi[s_idx] = 0.0
        else:
            R_pi[s_idx] = R[s_idx, policy[s_idx]]
    print(R_pi)
    # Solve the Bellman equation: V = (I - γP)^(-1) * R_pi
    V_pi = np.linalg.inv(np.eye(n) - gamma * P) @ R_pi
    return V_pi


print("Optimal Policy (state index -> action index):\n", POLICY)

old_old_policy = POLICY.copy()
while True:
    old_policy = POLICY.copy()
    P_pi = p_wrt_policy(Psas, POLICY)
    #set gamma to 0.9 for value computation so it takes into account future rewards as well as tap fills
    #previously I think that having it set close to 1 (0.99) made it equally prioritize two different actions
    #and so the POLICY kept oscillating between them, never converging
    #and so never terminating the loop
    #NEVERMIND IT STILL DOES IT
    V = value_computation(P_pi, R_MATRIX, POLICY, gamma=0.99)
    #print("Value function V:\n", V)
    gamma = 0.99
    
    Q = np.zeros_like(R_MATRIX)
    for s in range(num_states):
        for a in range(num_actions):
            total = 0.0
            for n in range(num_states):
                total += Psas[s, a, n] * V[n]
                # print("state: " string(n) " value: " string(V[n]))
            Q[s, a] = R_MATRIX[s, a] + gamma * total
    
    for i in range(num_states):
        s_idx = i
        if terminal_flags[s_idx]:
            POLICY[s_idx] = None
        else:
            # old_old_policy = old_policy.copy()
            # old_policy = POLICY.copy()
            POLICY[s_idx] = int(np.argmax(Q[s_idx]))
            print(Q[s_idx])
    print("Optimal Policy (state index -> action index):\n", POLICY)
    if POLICY == old_policy:
        break

