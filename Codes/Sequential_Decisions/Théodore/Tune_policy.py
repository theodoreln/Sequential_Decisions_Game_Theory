######### Tune policy parameters #########

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import gurobipy as gb
from gurobipy import GRB

from Valuefunction_LP import Policy_LP
from Valuefunction_DP import Policy_DP
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
        
Best_dec = Policy_LP(n, base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba)


# Define the model
model = gb.Model("work_study_model") 

# Set time limit
model.Params.TimeLimit = 100

# Add variables 
alpha = model.addVar(vtype = GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Variable alpha")
beta = model.addVar(vtype = GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Variable beta")
gamma = model.addVar(vtype = GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Variable gamma")

# Set objective function
objective = gb.quicksum(gb.quicksum(((alpha - beta * j - gamma * i) - Best_dec[i,j])**2 for j in range(base_education, max_education+1)) for i in range(n))
model.setObjective(objective, GRB.MINIMIZE)

# Optimization
model.optimize()

# Extract result of the optimization
alpha_opt = alpha.X
beta_opt = beta.X
gamma_opt = gamma.X

print("Optimal alpha: ", alpha_opt)
print("Optimal beta: ", beta_opt)
print("Optimal gamma: ", gamma_opt)

# Extract the tuned policy decision for each period and education level
Best_dec_tuned = np.zeros((n, max_education+1-base_education))
for i in range(n) :
    for j in range(base_education, max_education+1) :
        Best_dec_tuned[i,j] = round(alpha_opt - beta_opt * j - gamma_opt * i, 0)
print(Best_dec_tuned)        

    
# Plotting function for two Best_dec matrices
def Plotting(Best_dec1, Best_dec2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first Best_dec matrix
    data1 = Best_dec1.T
    cmap = ListedColormap(['blue', 'green'])
    im1 = axs[0].imshow(data1, cmap=cmap, aspect='auto')
    axs[0].set_xticks(np.arange(data1.shape[1]))
    axs[0].set_yticks(np.arange(data1.shape[0]))
    axs[0].set_xticklabels(np.arange(1, data1.shape[1] + 1))
    axs[0].set_yticklabels(np.arange(1, data1.shape[0] + 1))
    axs[0].set_xticks(np.arange(-0.5, data1.shape[1], 1), minor=True)
    axs[0].set_yticks(np.arange(-0.5, data1.shape[0], 1), minor=True)
    axs[0].grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Education level")
    axs[0].set_title("Work / Study Optimal")

    # Plot the second Best_dec matrix
    data2 = Best_dec2.T
    im2 = axs[1].imshow(data2, cmap=cmap, aspect='auto')
    axs[1].set_xticks(np.arange(data2.shape[1]))
    axs[1].set_yticks(np.arange(data2.shape[0]))
    axs[1].set_xticklabels(np.arange(1, data2.shape[1] + 1))
    axs[1].set_yticklabels(np.arange(1, data2.shape[0] + 1))
    axs[1].set_xticks(np.arange(-0.5, data2.shape[1], 1), minor=True)
    axs[1].set_yticks(np.arange(-0.5, data2.shape[0], 1), minor=True)
    axs[1].grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Education level")
    axs[1].set_title("Work / Study Tuned and Rounded")

    plt.tight_layout()
    plt.show()

Plotting(Best_dec, Best_dec_tuned)
        