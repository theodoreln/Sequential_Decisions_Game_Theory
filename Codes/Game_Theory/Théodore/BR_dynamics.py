import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import GRB

# Define constants
numPlayers = 10
base_salary = 10
education_cost_coefficient = 15
c = 10
d = 1

# Definition of base functions

def calculate_average_education(education_levels):
    return np.mean(education_levels)

def calculate_sum_education(education_levels) :
    return np.sum(education_levels)

def salary_1(player_education, avg_education):
    return base_salary * (1 + player_education / avg_education)

def salary_1_continuous(player_education, inv_avg_education):
    return base_salary * (1 + player_education * inv_avg_education)

def salary_2(player_education, total_education) :
    return base_salary * (c + player_education * player_education - d * total_education)

def cost_of_education(education_level):
    return education_cost_coefficient * education_level

def money(player_education, avg_education, total_education, salary_version):
    if salary_version == 1 :
        return salary_1(player_education, avg_education) - cost_of_education(player_education)
    elif salary_version == 2 :
        return salary_2(player_education, total_education) - cost_of_education(player_education)

# Function to compute the discrete best response for a player given others' choices
def best_response_discrete(player_index, education_levels, type, salary_version):
    avg_education = calculate_average_education(education_levels)
    total_education = calculate_sum_education(education_levels)
    best_education = 0
    best_payoff = -np.inf
    
    # Condition on binary or discrete 
    if type == "binary" : 
        rg = [0, 1]
    elif type == "discrete" :
        rg = np.linspace(0, 1, 101)
    
    # Test every value of education to seek for the best
    for e in rg: 
        temp_education_levels = education_levels.copy()
        temp_education_levels[player_index] = e
        avg_education = calculate_average_education(temp_education_levels)
        total_education = calculate_sum_education(temp_education_levels)
        payoff = money(e, avg_education, total_education, salary_version)
        if payoff > best_payoff:
            best_education = e
            best_payoff = payoff
    
    return best_education


# Function to compute the discrete best response for a player given others' choices
def best_response_continuous(player_index, education_levels, salary_version):
    
    temp_education_levels = education_levels.copy()
    
    model = gb.Model("work_study_model") 
    
    # Set time limit
    model.Params.TimeLimit = 100
    
    # Add variables 
    education = model.addVar(vtype = GRB.CONTINUOUS,
                             lb = 0,
                             ub = 1,
                             name="Action of player"
                             )
    
    if salary_version == 1 :
        avg_education = model.addVar(vtype = GRB.CONTINUOUS,
                                    lb = 0,
                                    ub = 1,
                                    name="Action of player"
                                    )
        # To solve the problem of the division by a variable
        inv_avg_education = model.addVar(vtype = GRB.CONTINUOUS,
                                         lb = 1,
                                         ub = gb.GRB.INFINITY,
                                         name="Inverse of the average salary"
                                         )
    
    if salary_version == 2 :
        total_education = model.addVar(vtype = GRB.CONTINUOUS,
                                    lb = 0,
                                    ub = gb.GRB.INFINITY,
                                    name="Action of player"
                                    )
    
    sal = model.addVar(vtype = GRB.CONTINUOUS,
                       lb=-gb.GRB.INFINITY, 
                       ub=gb.GRB.INFINITY,
                       name="Salary of player"
                       )
    
    # Set objective function
    objective = sal  - education_cost_coefficient*education
    model.setObjective(objective, GRB.MAXIMIZE)
    
    
    if salary_version == 1 :
        avg_constraint = model.addConstr(avg_education, 
                                            gb.GRB.EQUAL,
                                            calculate_average_education(np.concatenate((temp_education_levels[:player_index], np.array([education]), temp_education_levels[player_index+1:]))),
                                            name="Avg Salary definition of player"
                                            )
        
        inv_avg_constraint = model.addConstr(avg_education * inv_avg_education, 
                                            gb.GRB.EQUAL,
                                            1,
                                            name="Avg Salary definition of player"
                                            )
        
        salary_constraint = model.addConstr(sal, 
                                            gb.GRB.EQUAL,
                                            salary_1_continuous(education,inv_avg_education),
                                            name="Salary definition of player"
                                            )
        
    elif salary_version == 2 :
        total_constraint = model.addConstr(total_education, 
                                        gb.GRB.EQUAL,
                                        calculate_sum_education(np.concatenate((temp_education_levels[:player_index], np.array([education]), temp_education_levels[player_index+1:]))),
                                        name="Total Salary definition of player"
                                        )
        
        salary_constraint = model.addConstr(sal, 
                                            gb.GRB.EQUAL,
                                            salary_2(education,total_education),
                                            name="Salary definition of player"
                                            )
    
    model.setParam('NonConvex', 2)
    
    model.optimize()
    
    best_education = education.x
    
    return best_education


# Best Response Dynamics loop
def best_response_dynamics(iterations, type, salary_version):
    
    # Initialization
    if type == 'binary' :
        education_levels = np.random.choice([0, 1], size=numPlayers)
    if type == 'discrete' :
        education_levels = np.round(np.random.uniform(0, 1, numPlayers), 2)
    if type == 'continuous' :
        education_levels = np.random.uniform(0, 1, numPlayers)
        
    # Keep history
    history = np.zeros((iterations+1, numPlayers))
    history[0, :] = education_levels  # Store initial values
    
    # Iteration for convergence
    for t in range(iterations):
        for player in range(numPlayers):
            if type == 'binary' or type == "discrete" :
                education_levels[player] = best_response_discrete(player, education_levels, type, salary_version)
            elif type == 'continuous' :
                education_levels[player] = best_response_continuous(player, education_levels, salary_version)
        history[t+1, :] = education_levels  # Store education levels at each iteration
        print(f"Average Education after iteration: {calculate_average_education(education_levels)}")
    return history



# Run the dynamics
iterations = 25
history = best_response_dynamics(iterations, "continuous", 2)
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
