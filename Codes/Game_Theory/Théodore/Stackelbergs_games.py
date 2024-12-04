import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import GRB

# Define constants
numPlayers = 10
players = [i for i in range(numPlayers)]
base_salary = 10
education_cost_coefficient = {0:8, 1:8, 2:8, 3:8, 4:8, 5:12, 6:8, 7:8, 8:8, 9:8}
Budget = 5
c = 1
d = 0.1


# Function to compute the leader's problem
def stackelberg():
    
    model = gb.Model("work_study_model") 
    
    # Set time limit
    model.Params.TimeLimit = 100
    
    # Add variables 
    education = {i: model.addVar(vtype = GRB.CONTINUOUS,
                                 lb = 0,
                                 ub = 1,
                                 name=f"Action of player {i}"
                                 ) for i in players}
    
    slack_1 = {i: model.addVar(vtype = GRB.CONTINUOUS,
                               lb = 0,
                               ub = gb.GRB.INFINITY,
                               name=f"Slack 1 of player {i}"
                               ) for i in players}
    
    slack_2 = {i: model.addVar(vtype = GRB.CONTINUOUS,
                               lb = 0,
                               ub = gb.GRB.INFINITY,
                               name=f"Slack 2 of player {i}"
                               ) for i in players}
    
    subsidy = {i: model.addVar(vtype = GRB.CONTINUOUS,
                               lb = 0,
                               ub = gb.GRB.INFINITY,
                               name=f"Subsidy of player {i}"
                               ) for i in players}
    
    min_ed = model.addVar(vtype = GRB.CONTINUOUS,
                               lb = 0,
                               ub = gb.GRB.INFINITY,
                               name=f"minimum education"
                               )
    
    # Set objective function maximize total education
    objective = gb.quicksum(education[i] for i in players)
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Set objective function maximize minimum education
    # model.setObjective(min_ed, GRB.MAXIMIZE)
    
    Budget_constraint = model.addConstr(gb.quicksum(subsidy[i] for i in players), 
                                            gb.GRB.LESS_EQUAL,
                                            Budget,
                                            name="Limit on budget"
                                            )
    
    Slack_1_constraint = {i: model.addQConstr(slack_1[i] * (education[i] - 1), 
                                            gb.GRB.EQUAL,
                                            0,
                                            name=f"Slack 1 constraint for player {i}"
                                            ) for i in players}
    
    Slack_2_constraint = {i: model.addQConstr(slack_2[i] * (-education[i]), 
                                            gb.GRB.EQUAL,
                                            0,
                                            name=f"Slack 2 constraint for player {i}"
                                            ) for i in players}
    
    Lagrange_constraint = {i: model.addQConstr(base_salary * (d-1) + (education_cost_coefficient[i]-subsidy[i]) + slack_1[i] -slack_2[i], 
                                            gb.GRB.EQUAL,
                                            0,
                                            name=f"Lagrange constraint for player {i}"
                                            ) for i in players}
    
    minimum_constraint = {i: model.addConstr(min_ed, gb.GRB.LESS_EQUAL, education[i], name=f"Minimum education for player {i}") for i in players}
    
    model.optimize()
    
    education = [education[i].x for i in players]
    subsidy = [round(subsidy[i].x,2) for i in players]
    slack_1 = [slack_1[i].x for i in players]
    slack_2 = [slack_2[i].x for i in players]
    
    return education, subsidy, slack_1, slack_2



education, subsidy, slack_1, slack_2 = stackelberg()
print("Education : ", education)
print("Subsidy : ", subsidy)
print(slack_1)
print(slack_2)