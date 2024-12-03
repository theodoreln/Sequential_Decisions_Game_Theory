import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import GRB

# Define constants
numPlayers = 10
base_salary = 10
education_cost_coefficient = 3
c = 1
d = 0.05

# List of players from 1 to numPlayers
Players = list(range(1, numPlayers + 1))

    
# Optimization function to solve the work vs educate game with continuous actions

def continuous_actions(salary_version) :
    
    model = gb.Model("work_educate")
    
    # Create variables
    education_levels = {i: model.addVar(vtype=GRB.CONTINUOUS,
                                    lb = 0,
                                    ub = 1, 
                                    name=f"Education of player {i}") for i in Players}
    
    # if salary_version == 1 :
    #     avg_education = model.addVar(vtype = GRB.CONTINUOUS,
    #                                 lb = 0,
    #                                 ub = 1,
    #                                 name="Action of player"
    #                                 )
    #     # To solve the problem of the division by a variable
    #     inv_avg_education = model.addVar(vtype = GRB.CONTINUOUS,
    #                                      lb = 1,
    #                                      ub = gb.GRB.INFINITY,
    #                                      name="Inverse of the average salary"
    #                                      )
    
    if salary_version == 2 :
        total_education = model.addVar(vtype = GRB.CONTINUOUS,
                                    lb = 0,
                                    ub = gb.GRB.INFINITY,
                                    name="Sum of Action of player"
                                    )
    
    salary = {i: model.addVar(vtype=GRB.CONTINUOUS,
                                    lb = -gb.GRB.INFINITY,
                                    ub = gb.GRB.INFINITY, 
                                    name=f"Salary of player {i}") for i in Players}
    
    money = {i: model.addVar(vtype=GRB.CONTINUOUS,
                                    lb = -gb.GRB.INFINITY,
                                    ub = gb.GRB.INFINITY, 
                                    name=f"Money of player {i}") for i in Players}
    
    s = model.addVar(vtype = GRB.CONTINUOUS,
                     lb = -gb.GRB.INFINITY,
                     ub = gb.GRB.INFINITY,
                     name="Minimum money"
                     )
    
    # Set objective function 
    objective = gb.quicksum(money[i] for i in Players)
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Add constraints
    # if salary_version == 1 :
    #     avg_constraint = model.addConstr(avg_education, 
    #                                         gb.GRB.EQUAL,
    #                                         gb.quicksum(education_levels[i] for i in Players)/numPlayers,
    #                                         name="Avg Salary definition of players"
    #                                         )
        
    #     inv_avg_constraint = model.addConstr(avg_education * inv_avg_education, 
    #                                         gb.GRB.EQUAL,
    #                                         1,
    #                                         name="Inv Avg Salary definition of players"
    #                                         )
        
    #     for i in Players :
    #         model.addConstr(salary[i], 
    #                         gb.GRB.EQUAL,
    #                         base_salary * (1 + education_levels[i] * inv_avg_education),
    #                         name=f"Salary definition of player {i}"
    #                         )
            
    #         model.addConstr(money[i],
    #                         gb.GRB.EQUAL,
    #                         salary[i] - education_cost_coefficient * education_levels[i],
    #                         name=f"Money definition of player {i}"
    #                         )
        
    if salary_version == 2 :
        total_constraint = model.addConstr(total_education, 
                                           gb.GRB.EQUAL,
                                           gb.quicksum(education_levels[i] for i in Players),
                                           name="Total Salary definition of player"
                                           )
        
        for i in Players :
            model.addConstr(salary[i], 
                            gb.GRB.EQUAL,
                            base_salary*(c+education_levels[i]-d*total_education),
                            name=f"Salary definition of player {i}"
                            )
            
            model.addConstr(money[i],
                            gb.GRB.EQUAL,
                            salary[i] - education_cost_coefficient * education_levels[i],
                            name=f"Money definition of player {i}"
                            )
            
            model.addConstr(s, 
                            gb.GRB.LESS_EQUAL,
                            money[i],
                            name=f"Minimum salary definition of player {i}"
                            )
        
    model.setParam('NonConvex', 2)
    
    model.optimize()
    
    # Get education values
    best_education = {i: education_levels[i].x for i in Players}
    
    # Print salaries
    for i in Players :
        print(f"Salary of player {i}: {salary[i].x}")
    
    # Print money
    for i in Players :
        print(f"Money of player {i}: {money[i].x}")
        
    # Print total education
    if salary_version == 2 :
        print(f"Total education: {total_education.x}")
    
    # # Print average education
    # if salary_version == 1 :
    #     print(f"Average education: {avg_education.x}")
    
    # # Print inv average education 
    # if salary_version == 1 :
    #     print(f"Inv average education: {inv_avg_education.x}")
    
    return best_education

# Call the function
education = continuous_actions(2)

for i in Players :
    print(f"Player {i} will educate {education[i]}")