######### Approximate Dynamic Programming #########

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
gamma1 = 0.95

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

# Function giving the approximate value function
def Approx_V(education, time, alpha, beta, gamma2) :
    return alpha * education + beta * time + gamma2 * education * time

# Calculate Bellman error
def bellman_error(edu_now, time, Return, edu_new, alpha, beta, gamma2) :
    return (Return + gamma1 * Approx_V(edu_new, time+1, alpha, beta, gamma2) - Approx_V(edu_now, time, alpha, beta, gamma2))

# Recursion to tune the parameters alpha, beta, gamma
lambda1 = 0.0000005
threshold = 10e-12
alpha = 1
beta = 1
gamma2 = 1
i = 0
while True :
    i += 1
    print("Iteration number : ", i)
    # Choose a random education, time and decision (can't be last time step or 2 last educations)
    edu_now = random.randint(base_education, max_education)
    time = random.randint(1, n)
    print("education = ", edu_now, ", time = ", time)
    V_study = Approx_V(edu_now + education_rate, time+1, alpha, beta, gamma2)
    V_work = Approx_V(edu_now, time+1, alpha, beta, gamma2)
    print("Value function study : ", V_study, ", Value function work : ", V_work)
    if V_study > V_work :
        dec_now = 1
    else :
        dec_now = 0
    print("decision = ", dec_now)
    edu_new = edu_now + dec_now * education_rate
    Return = ret(edu_now, dec_now)
    
    print("Value function now : ", Approx_V(edu_now, time, alpha, beta, gamma2), ", Value function new : ", Return + gamma1*Approx_V(edu_new, time+1, alpha, beta, gamma2))

    # Calculate the gradient of the Bellman error
    error = bellman_error(edu_now, time, Return, edu_new, alpha, beta, gamma2)
    print("Error : ", error)
    
    # Gradient
    grad_alpha = 2*(gamma1*edu_new - edu_now)*error
    grad_beta = 2*(gamma1*(time+1)-time)*error
    grad_gamma2 = 2*(gamma1*edu_new*(time+1) - edu_now*time)*error
    # New parameters
    new_alpha = alpha - lambda1 * grad_alpha
    new_beta = beta - lambda1 * grad_beta
    new_gamma2 = gamma2 - lambda1 * grad_gamma2
    
    # Check for convergence
    if i>=10000:
        break
    # Update the parameters
    alpha = new_alpha
    beta = new_beta
    gamma2 = new_gamma2
    print("alpha = ", alpha, ", beta = ", beta, ", gamma = ", gamma2)
    
print("Number of iterations: ", i)
print("Optimal alpha: ", alpha)
print("Optimal beta: ", beta)
print("Optimal gamma: ", gamma2)

# Define the policy function
def policy(education, period):
    # Let us compute Q for all the actions possibles (work or study)  
    # the action is work
    Q_w = - expenses + base_salary * (1 + education/2) + gamma1 * Approx_V(period+1, education, alpha, beta, gamma2)
    # the action is study
    Q_s = - expenses + gamma1 * Approx_V(period+1, education+2, alpha, beta, gamma2)

    if Q_w > Q_s : # the action to take is work
        return 0
    else : # the action to take is study
        return 1
    
# Fill out the Best_dec array
Best_dec = np.zeros((n, max_education+1-base_education))
for i in range(n) :
    for j in range(base_education, max_education+1) :
        Best_dec[i,j-base_education] = policy(j, i+1)
        
print(Best_dec)
        
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
        
Plotting(Best_dec)