#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:03:43 2024

@author: salomeaubri
"""

import gurobipy as gb
from gurobipy import GRB
import numpy as np
import random

number_of_players = 10
PLAYERS = [_ for _ in range(number_of_players)]
base_salary = 2
education_cost = 1
c = 1
d=0.25


players_action = [random.randint(0, 1) for i in range(number_of_players)]
print("Initial actions", players_action)

"""
# t=0
education = [random.randint(0, 1) for i in range(number_of_players)]
payoff_0 = [] 
for i in range(number_of_players):
    s = base_salary * (c + education[i] - d*sum(education))
    payoff_0.append(s-c)
payoff.append(payoff_0)
print(education)
print(payoff)
"""

obj = {-1:0, 0:1}
k = 0
threshold = 0.001

while obj[k]-obj[k-1] > threshold:
    k+=1

    model = gb.Model("work_study_model") 
    
    # Set time limit
    model.Params.TimeLimit = 100
    
    # Add variables 
    action = {
        i: model.addVar(
             vtype = GRB.BINARY, #0 is work, 1 is study
             name="Action of player {0}".format(i)
             ) for i in PLAYERS}
    
    salary = {
        i: model.addVar(
             lb=-gb.GRB.INFINITY, 
             ub=gb.GRB.INFINITY,
             name="Salary of player {0}".format(i)
             ) for i in PLAYERS}
    
    # Set objective function
    objective = gb.quicksum(salary[i] - education_cost*action[i] for i in PLAYERS)
    model.setObjective(objective, GRB.MINIMIZE)
    
    salary_constraint = {
        i : model.addConstr(  
            salary[i],
            gb.GRB.EQUAL,
            base_salary * (c + action[i] - d * gb.quicksum(players_action[j] for j in PLAYERS if j != i)),
            name="Salary definition of player {0}".format(i)
        ) for i in PLAYERS}
    
    model.optimize()
    
    for v in model.getVars():
        print('%s %g' % (v.VarName, v.X))
    
    print('Objective function: %g' % model.ObjVal)
    
    players_action = [action[i].x for i in range(number_of_players)]
    obj[k] = model.ObjVal
    

print("Nash equilibrium", players_action)
print("number of optimization", k)
print("objective function", obj[k])








