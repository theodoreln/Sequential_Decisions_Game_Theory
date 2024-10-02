import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import GRB

# Define constants
numPlayers = 10
base_salary = 10
education_cost_coefficient = 15

# Initialize random education levels for all players #

### Every values between 0 and 10 with step 0.1
# education_levels = np.round(np.random.uniform(0, 10, numPlayers), 1)

### Every values between 0 and 1, continuous
education_levels = np.random.uniform(0, 1, numPlayers)

### Binary possibility, 0 or 1
# education_levels = np.random.choice([0, 1], size=numPlayers)

def calculate_average_education(education_levels):
    return np.mean(education_levels)

def salary(player_education, avg_education):
    return base_salary * (1 + player_education / avg_education)

def cost_of_education(education_level):
    return education_cost_coefficient * education_level

def money(player_education, avg_education):
    return salary(player_education, avg_education) - cost_of_education(player_education)

# Function to compute the discrete best response for a player given others' choices
def best_response_discrete(player_index, education_levels):
    avg_education = calculate_average_education(education_levels)
    best_education = 0
    best_payoff = -np.inf
    
    # Try different levels of education to find the best response #
    
    ### Assuming education levels between 0 and 10 with step 0.1
    # for e in np.linspace(0, 10, 101): 
        
    ### Assuming education levels between 0 and 1 with step 0.001
    for e in np.linspace(0, 1, 1001): 
    
    ### Assuming education levels between 0 and 1 
    # for e in [0, 1]: 
        temp_education_levels = education_levels.copy()
        temp_education_levels[player_index] = e
        avg_education = calculate_average_education(temp_education_levels)
        payoff = money(e, avg_education)
        if payoff > best_payoff:
            best_education = e
            best_payoff = payoff
    
    return best_education


# Function to compute the discrete best response for a player given others' choices
def best_response_continuous(player_index, education_levels):
    
    model = gb.Model("work_study_model") 
    
    # Set time limit
    model.Params.TimeLimit = 100
    
    # Add variables 
    education = model.addVar(vtype = GRB.CONTINUOUS,
                             lb = 0,
                             ub = 1,
                             name="Action of player"
                             )
    
    avg_education = model.addVar(vtype = GRB.CONTINUOUS,
                                 lb = 0,
                                 ub = 1,
                                 name="Action of player"
                                 )
    
    sal = model.addVar(vtype = GRB.CONTINUOUS,
                       lb=-gb.GRB.INFINITY, 
                       ub=gb.GRB.INFINITY,
                       name="Salary of player"
                       )
    
    # Set objective function
    objective = sal - education_cost_coefficient*education
    model.setObjective(objective, GRB.MAXIMIZE)
    
    avg_constraint = model.addConstr(avg_education, 
                                     gb.GRB.EQUAL,
                                     calculate_average_education(education_levels[:player_index] + [education] + education_levels[player_index+1:]),
                                     name="Avg definition of player"
                                     )
    
    salary_constraint = model.addConstr(sal, 
                                        gb.GRB.EQUAL,
                                        salary(education,avg_education),
                                        name="Salary definition of player"
                                        )
    
    model.setParam('NonConvex', 2)
    
    model.optimize()
    
    best_education = education.x
    
    return best_education


# Best Response Dynamics loop
def best_response_dynamics(education_levels, iterations):
    history = np.zeros((iterations+1, numPlayers))
    history[0, :] = education_levels  # Store initial values
    
    for t in range(iterations):
        for player in range(numPlayers):
            education_levels[player] = best_response_continuous(player, education_levels)
        history[t+1, :] = education_levels  # Store education levels at each iteration
        print(f"Average Education after iteration: {calculate_average_education(education_levels)}")
    return history

# Run the dynamics
iterations = 25
history = best_response_dynamics(education_levels, iterations)
print("Final education levels:", history[iterations,:])
print(history)

# Plot the education levels over iterations for each player
plt.figure(figsize=(10, 6))
for player in range(numPlayers):
    plt.plot(range(iterations+1), history[:, player], label=f'Player {player+1}')

# Customize plot
plt.xlabel('Iterations')
plt.ylabel('Education Level')
plt.title('Education Levels Over Iterations for Each Player')
plt.legend(loc='best')
plt.grid(False)  # Disable the grid
plt.show()
