#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:35:39 2024

@author: salomeaubri
"""
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import policy


number_of_periods = 10
T = range(0, number_of_periods)

# Parameters
expenses = 1
base_salary = 4
education_rate = 2 # has to be an integer

gamma = 0.1

max_education = number_of_periods * education_rate

# Dictionary to store the value function
store_value = {}

def value_function(x,t): # x represents the current education level and t the current period
    if (t,x) in store_value :
        return store_value[t,x]
    
    else :
        if t == number_of_periods:
            # work is the best solution
            V = - expenses + 1 * base_salary * (1 + x/2)
            work = 1
            study = 0
        
        else:
            V_study = - expenses + value_function(x + 1 * education_rate, t+1)
            V_work = - expenses + base_salary * (1 + x/2) + value_function(x, t+1)
            if V_study > V_work :
                V = V_study
                study = 1
                work = 0
            else:
                V = V_work
                study = 0
                work = 1
        return V

for t in range(number_of_periods+1):
    for x in range(0, max_education+1):
        store_value[t,x] = value_function(x, t)

def policy(education, period):
    # Let us compute Q for all the actions possibles (work or study)  
    # the action is work
    Q_w = - expenses + base_salary * (1 + education/2) + gamma * sum(value_function(potential_education, next_t) for potential_education in range(int(education+1), max_education + 1) for next_t in range(period+1, number_of_periods+1))
    # the action is study
    Q_s = - expenses + gamma * sum(value_function(potential_education, next_t) for potential_education in range(int(education), max_education + 1) for next_t in range(period+1, number_of_periods+1))

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
    education[period] = education[period - 1] + study[period-1] * education_rate
    work[period], study[period] = policy(education[period], period)
    if (work[period] + study[period] > 1) or (work[period] != 0 and work[period] != 1) or (study[period] != 0 and study[period] != 1):
        work[period], study[period] = 0, 0
    print(period)
    print(work[period], study[period], money[period], education[period])
        

plt.plot(T, work, label='work', marker='o')
plt.plot(T, study, label='study', marker='s')
plt.plot(T, money, label='money', marker='^')
plt.plot(T, education, label='education', marker='x')

# Add labels and title
plt.xlabel('time')
#plt.ylabel('Y-axis label')
#plt.title('Plot of Four 5-Element Arrays')
plt.legend()  # Show the legend

# Show the plot
plt.show()



