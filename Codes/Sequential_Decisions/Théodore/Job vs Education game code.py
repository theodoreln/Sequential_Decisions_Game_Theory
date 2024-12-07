############# Main code to launch the Job vs Education game ##############

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

from Valuefunction_LP import Policy_LP
from Valuefunction_DP import Policy_DP
from Value_iteration import Policy_Value_iteration
from Policy_iteration import Policy_Policy_iteration
from Tune_policy import Policy_Function_Approximation


### Parameters 
# Numbers of periods
n = 10
# Base salary
base_salary = 1
# Base Education
base_education = 0
# Base Money
base_money = 3
# Expenses
expenses = 1
# Education Rate
education_rate = 2
# Max education
max_education = base_education + n * education_rate
# Discount factor
gamma = 0.95

# Return function at a stage x for an action u 
def ret(edu_now, dec_now) :
    return ((1 - dec_now) * (base_salary * (1 + edu_now / 2)) - expenses)

# Probability function of arriving to stage x' with choosing action u in stage x
def proba(edu_new, edu_now, dec_now) :
    # Difference between current and old education
    diff_edu = edu_new - edu_now
    # If we choose to work, we are sure to stay at the same education
    if dec_now == 0 :
        if diff_edu == 0 :
            return 1
        else :
            return 0
        # If we choose to study, we have a chance to improve by 2,1, or 0 our education level
    elif dec_now == 1 :
        if diff_edu == education_rate :
            return 0.7
        elif diff_edu == 0 :
            return 0.3
        else :
            return 0
        
#Plotting
def Plotting(Best_dec) :
    # Transpose the data to have the first dimension on the x-axis
    data = Best_dec.T
    # Create a colormap for 0 -> blue and 1 -> green
    cmap = ListedColormap(['blue', 'green'])
    # Create the plot
    plt.imshow(data, cmap=cmap, aspect='auto')
    # Set the ticks for both x and y axes to be integers
    plt.xticks(np.arange(data.shape[1]), np.arange(1, data.shape[1]+1))
    plt.yticks(np.arange(data.shape[0]), np.arange(base_education, base_education + data.shape[0]))
    # Add grid lines (around each square)
    plt.gca().set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    # Invert the y-axis so it matches your plot style
    plt.gca().invert_yaxis()
    # Add labels for x and y axes
    plt.xlabel("Time")
    plt.ylabel("Education level")
    # Add a title
    plt.title("Work / Study Decision")
    # Show the plot
    plt.show()

T = range(n)

money = np.zeros(n)
education = np.zeros(n)
work = np.zeros(n)
study = np.zeros(n)

money[0] = base_money
education[0] = base_education

Best_dec = Policy_Function_Approximation(n, base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba)
Plotting(Best_dec)

money_care = True

for period in range(n):
    
    Decision = Best_dec[period, int(education[period])-base_education]
    
    if Decision == 0 :
        work[period] = 1
        study[period] = 0
    elif Decision == 1 :
        work[period] = 0
        study[period] = 1
    
    if period != n-1 :
        money[period+1] = money[period] + ret(education[period], Decision)
        if money_care :
            if money[period+1] < 0 :
                work[period] = 1
                study[period] = 0
                Decision = 0
                money[period+1] = money[period] + ret(education[period], 0)
        education[period+1] = np.random.choice([l for l in range(int(education[period]), max_education+1)], p=[proba(l, education[period], Decision) for l in range(int(education[period]), max_education+1)])

plt.plot(T, work, label='work', marker='o')
plt.plot(T, study, label='study', marker='s')
plt.plot(T, money, label='money', marker='^')
plt.plot(T, education, label='education', marker='x')

# Add labels and title
plt.xlabel('time')
plt.legend()  # Show the legend
# Time in x ticks from 1 to n
plt.xticks(np.arange(n), np.arange(1, n+1))
# Show the plot
plt.show()
