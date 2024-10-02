############# Main code to launch the Job vs Education game ##############

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import policy

from Valuefunction_LP import Policy_LP
from Value_iteration import Policy_Value_iteration
from Policy_iteration import Policy_Policy_iteration


### Parameters 
# Numbers of periods
n = 20
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
max_education = base_education + n * 2
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
        

T = range(n)

money = np.zeros(n)
education = np.zeros(n)
work = np.zeros(n)
study = np.zeros(n)

money[0] = base_money
education[0] = base_education

Best_dec = Policy_LP(n, base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba)
print(Best_dec)

for period in range(n):
    
    Decision = Best_dec[period, int(education[period])]
    
    if Decision == 0 :
        work[period] = 1
        study[period] = 0
    elif Decision == 1 :
        work[period] = 0
        study[period] = 1
    
    if period == 0 :
        money[period] = base_money + ret(education[period], Decision)
        if money[period] < 0 :
            work[period] = 1
            study[period] = 0
            Decision = 0
            money[period] = base_money + ret(education[period], 0)
        education[period+1] = np.random.choice([l for l in range(int(education[period]), max_education+1)], p=[proba(l, education[period], Decision) for l in range(int(education[period]), max_education+1)])
    elif period != 0 :
        money[period] = money[period-1] + ret(education[period], Decision)
        if money[period] < 0 :
            work[period] = 1
            study[period] = 0
            Decision = 0
            money[period] = money[period-1] + ret(education[period], 0)
        if period != n-1 :
            education[period+1] = np.random.choice([l for l in range(int(education[period]), max_education+1)], p=[proba(l, education[period], Decision) for l in range(int(education[period]), max_education+1)])


plt.plot(T, work, label='work', marker='o')
plt.plot(T, study, label='study', marker='s')
plt.plot(T, money, label='money', marker='^')
plt.plot(T, education, label='education', marker='x')

# Add labels and title
plt.xlabel('time')
plt.legend()  # Show the legend

# Show the plot
plt.show()
