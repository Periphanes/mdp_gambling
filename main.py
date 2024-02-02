# Initial Code taken from https://gist.github.com/tuxdna/7e29dd37300e308a80fc1559c343c545

#!/usr/bin/env python
# coding=utf-8
import numpy as np
import random
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)

"""
1: Procedure Policy_Iteration(S,A,P,R)
2:           Inputs
3:                     S is the set of all states
4:                     A is the set of all actions
5:                     P is state transition function specifying P(s'|s,a)
6:                     R is a reward function R(s,a,s')
7:           Output
8:                     optimal policy π
9:           Local
10:                     action array π[S]
11:                     Boolean variable noChange
12:                     real array V[S]
13:           set π arbitrarily
14:           repeat
15:                     noChange ←true
16:                     Solve V[s] = ∑s'∈S P(s'|s,π[s])(R(s,a,s')+γV[s'])
17:                     for each s∈S do
18:                               Let QBest=V[s]
19:                               for each a ∈A do
20:                                         Let Qsa=∑s'∈S P(s'|s,a)(R(s,a,s')+γV[s'])
21:                                         if (Qsa > QBest) then
22:                                                   π[s]←a
23:                                                   QBest ←Qsa
24:                                                   noChange ←false
25:           until noChange
26:           return π
"""


# 0.01 * 10 + 0.04 * 5 + 0.05 * 2 + 0.4 * 1 + 0.2 * 0.8 + 0.3 * 0 = 0.96

"""
0 - 0.01 prob for 10x
1 - 0.04 prob for 5x
2 - 0.05 prob for 2x
3 - 0.40 prob for 1.05x
4 - 0.20 prob for 0.8x
5 - 0.30 prob for 0x
6 ~ 19 - 1 mean state
"""

states = [i for i in range(500)]
actions = [i for i in range(100)]

N_STATES = len(states)
N_ACTIONS = len(actions)

P = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # transition probability
R = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # rewards


# Actions 0 ~ 4 encode actions of gambling
# Actions 5 ~ 99 encode actions of normal life with mean reward of 1, with Gaussian distribution


for i in range(N_STATES):
    for j in range(N_ACTIONS):
        if j < 5:
            gambling_states = random.sample(range(N_STATES), 6)
            for k in range(N_STATES):
                if k == gambling_states[0]:
                    P[i][j][k] = 0.01
                    R[i][j][k] = 10
                elif k == gambling_states[1]:
                    P[i][j][k] = 0.04
                    R[i][j][k] = 5
                elif k == gambling_states[2]:
                    P[i][j][k] = 0.05
                    R[i][j][k] = 2
                elif k == gambling_states[3]:
                    P[i][j][k] = 0.40
                    R[i][j][k] = 1.05
                elif k == gambling_states[4]:
                    P[i][j][k] = 0.20
                    R[i][j][k] = 0.8
                elif k == gambling_states[5]:
                    P[i][j][k] = 0.30
        
        else:
            rand_dist = np.random.rand(N_STATES)
            rand_dist = rand_dist / np.sum(rand_dist)
        
            P[i][j] = rand_dist

            rand_gaussian = np.random.normal(1, 0.1, N_STATES)
            R[i][j] = rand_gaussian

def simulate(policy):
    state = random.randint(0, N_STATES-1)
    simulation_iter = 1000
    gambling_count = 0

    for _ in range(simulation_iter):
        action = policy[state]
        new_state = np.random.choice(range(N_STATES), p=P[state][action])

        if action <= 5:
            gambling_count += 1    

        state = new_state

    return gambling_count

gamma = 0.80

# initialize policy and value arbitrarily
policy = [random.randint(0, N_ACTIONS - 1) for s in range(N_STATES)]
V = np.zeros(N_STATES)

print("Initialization Complete")

is_value_changed = True
iterations = 0

policy_iteration_count = 20

final_iteration_results = []

for _ in range(policy_iteration_count):
    is_value_changed = False
    iterations += 1
    for s in range(N_STATES):
        V[s] = sum([P[s,policy[s],s1] * (R[s,policy[s],s1] + gamma*V[s1]) for s1 in range(N_STATES)])

    for s in range(N_STATES):
        q_best = V[s]
        for a in range(N_ACTIONS):
            q_sa = sum([P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(N_STATES)])
            if q_sa > q_best:
                policy[s] = a
                q_best = q_sa
                is_value_changed = True

    print ("Iterations:", iterations)

    simulation_count = 100
    simulation_results = []
    
    for _ in range(simulation_count):
        simulation_results.append(simulate(policy))
    
    simulation_results = np.array(simulation_results)
    final_iteration_results.append(simulation_results)

print("Final policy")
print("policy")
print(V)

np_range = np.array([i for i in range(policy_iteration_count)])
mean = np.array([np.mean(i) for i in final_iteration_results])
std = np.array([np.std(i) for i in final_iteration_results])

plt.errorbar(np_range, mean, std, linestyle='None', marker='^')

plt.show()