#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:44:36 2024

@author: salomeaubri
"""

import gurobipy as gb
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

number_of_periods = 10
TIME = [_ for _ in range(1, number_of_periods+1)]

# Parameters
expenses = 1
base_salary = 1
education_rate = 2 # has to be an integer

Gamma = 1

max_education = number_of_periods * education_rate
STATES = [_ for _ in range(max_education+1)]

ACTIONS = [0, 1] # for convenience, 1 represents work and 0 study

## if you want to consider money, you can put the reward to - infinity if money goes under 0

def edu_proba(edu_diff): # to alleviate the problem, I code it not as a 3D matrix - maximum recursive depth reach otherwise
    if edu_diff == 0:
        return 0.3
    elif edu_diff == 1:
        return 0
    elif edu_diff == 2:
        return 0.7
    else:
        return 0

def random_education():
    return random.choices([0, 1, 2], weights=[0.3, 0, 0.7], k=1)[0]


#%% Value approximation with the gradient descent
alpha = 1
beta = 1
gamma = 1
desc_factor = 0.0005

threshold = 10e-4

i=0

def error(alpha, beta, gamma, x1, x2, t, R):
    return alpha * x1 + beta * t + gamma * x1 * t - (R + alpha * x2 + beta * (t+1) + gamma * x2 * (t+1))

while True:
    i+=1

    period = random.randint(0, number_of_periods)
    education = random.randint(0, max_education)
    action = random.randint(0,1) # if 1 then work
    reward = action * base_salary * (1 + education/2) - expenses
    education_next = education + random_education() 
    
    err = error(alpha, beta, gamma, education, education_next, period, reward)
    new_alpha = alpha - desc_factor * 2 * (education - Gamma * education_next) * err
    new_beta = beta - desc_factor * 2 * (period - Gamma * (period+1)) * err
    new_gamma = gamma - desc_factor * 2 * (education * period - Gamma * education_next * (period+1)) * err
    
    
    # Check for convergence
    if (abs(alpha - new_alpha) < threshold and abs(beta - new_beta) < threshold and abs(gamma - new_gamma) < threshold) or i>100:
        break
    alpha = new_alpha
    beta = new_beta
    gamma = new_gamma

print("Best alpha", alpha)
print("Best beta", beta)
print("Best gamma", gamma)

def V(period, education):
    return alpha * education + beta * period + gamma * education * period

#%% Policy

def policy(education, period):
    # Let us compute Q for all the actions possibles (work or study)  
    # the action is work
    Q_w = - expenses + base_salary * (1 + education/2) + Gamma * V(period+1, education)
    # the action is study
    Q_s = - expenses + Gamma * sum(edu_proba(variation) * V(period+1, education + variation) for variation in range(0,3))

    if Q_w > Q_s : # the action to take is work
        return 1, 0
    else : # the action to take is study
        return 0, 1
    

#%% Solution with the new policy

money = np.zeros(number_of_periods)
education = np.zeros(number_of_periods)
work = np.zeros(number_of_periods)
study = np.zeros(number_of_periods)

money[0] = 3
education[0] = 0

for period in range(1, number_of_periods):
    money[period] = money[period - 1] - expenses + work[period-1]*(base_salary)*(1+ education[period-1]/2)
    education[period] = education[period - 1] + study[period-1] * random_education()
    study[period], work[period] = policy(education[period], period)
    #if (work[period] + study[period] > 1) or (work[period] != 0 and work[period] != 1) or (study[period] != 0 and study[period] != 1):
    #    work[period], study[period] = 0, 0
    #print(period)
    #print(work[period], study[period], money[period], education[period])
        

plt.plot(TIME, work, label='work', marker='o')
plt.plot(TIME, study, label='study', marker='s')
plt.plot(TIME, money, label='money', marker='^')
plt.plot(TIME, education, label='education', marker='x')

# Add labels and title
plt.xlabel('time')
#plt.ylabel('Y-axis label')
#plt.title('Plot of Four 5-Element Arrays')
plt.legend()  # Show the legend

# Show the plot
plt.show()



