# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:21:43 2024

@author: geots
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import policy


#periods = np.array([1,2,3,4,5])
number_of_periods = 10
T = range(0, number_of_periods)

money = np.zeros(number_of_periods)
education = np.zeros(number_of_periods)
work = np.zeros(number_of_periods)
study = np.zeros(number_of_periods)

money[0] = 3
education[0] = 0

for period in range(1, number_of_periods):
    #expenses = random.choice([1,2])
    expenses = 1
    #base_salary = random.choice([2,3])
    base_salary = 1
    #education_rate = random.choice([2,3])
    education_rate = 2
    
    money[period] = money[period - 1] - expenses + work[period - 1]*(base_salary)*(1+ education[period - 1]/2)
    education[period] = education[period - 1] + study[period-1]*(education_rate)
    work[period], study[period] = policy.work_vs_study(money[period], education[period])
    if (work[period] + study[period] > 1) or (work[period] != 0 and work[period] != 1) or (study[period] != 0 and study[period] != 1):
        work[period], study[period] = 0, 0
        

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
