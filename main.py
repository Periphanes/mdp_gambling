# Initial Code taken from https://gist.github.com/tuxdna/7e29dd37300e308a80fc1559c343c545

#!/usr/bin/env python
# coding=utf-8
import numpy as np
import random

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
3 - 0.40 prob for 1x
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
                    R[i][j][k] = 1
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


gamma = 0.75

# initialize policy and value arbitrarily
policy = [random.randint(0, N_ACTIONS - 1) for s in range(N_STATES)]
V = np.zeros(N_STATES)

print("Initial policy", policy)
# print V
# print P
# print R

is_value_changed = True
iterations = 0
while is_value_changed:
    is_value_changed = False
    iterations += 1
    # run value iteration for each state
    for s in range(N_STATES):
        V[s] = sum([P[s,policy[s],s1] * (R[s,policy[s],s1] + gamma*V[s1]) for s1 in range(N_STATES)])
        # print "Run for state", s

    for s in range(N_STATES):
        q_best = V[s]
        # print "State", s, "q_best", q_best
        for a in range(N_ACTIONS):
            q_sa = sum([P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(N_STATES)])
            if q_sa > q_best:
                # print("State", s, ": q_sa", q_sa, "q_best", q_best)
                policy[s] = a
                q_best = q_sa
                is_value_changed = True

    print ("Iterations:", iterations)
    # print "Policy now", policy

print("Final policy")
print("policy")
print(V)