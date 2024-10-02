############## Deterministic optimization ##############

import gurobipy as gb
from gurobipy import GRB
import matplotlib.pyplot as plt

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

# Define the model
model = gb.Model("work_study_model") 
    
# Set time limit
model.Params.TimeLimit = 100

# Add variables 
work = {i: model.addVar(vtype = GRB.BINARY, name="Working at time {0}".format(i)) for i in range(n)}
study = {i: model.addVar(vtype = GRB.BINARY, name="Study at time {0}".format(i)) for i in range(n)}
education = {i: model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Education level at time {0}".format(i)) for i in range(n)}
money = {i: model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Money level at time {0}".format(i)) for i in range(n)}
profit = {i: model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Profit at time {0}".format(i)) for i in range(n)}

# Set objective function
objective = gb.quicksum(profit[i] - expenses for i in range(n))
model.setObjective(objective, GRB.MAXIMIZE)

# Constraints
salary_constraint = {i : model.addConstr(
    profit[i],
    gb.GRB.EQUAL,
    work[i] * (base_salary * (1 + (education[i] / 2))),
    name="Profit definition at time {0}".format(i)
    ) for i in range(n)}

Education_constraint = {i : model.addConstr(
    education[i],
    gb.GRB.EQUAL,
    education[i-1] + study[i-1] * education_rate,
    name="Education level definition at time {0}".format(i)
    ) for i in range(1,n)}
Education_constraint[0] = model.addConstr(education[0],gb.GRB.EQUAL,base_education,name="Education level definition at time 0")

Money_constraint = {i : model.addConstr(
    money[i],
    gb.GRB.EQUAL,
    money[i-1] + profit[i-1] - expenses,
    name="Money level definition at time {0}".format(i)
    ) for i in range(1,n)}
Money_constraint[0] = model.addConstr(money[0],gb.GRB.EQUAL,base_money,name="Money level definition at time 0")

Choice_constraint = {i : model.addConstr(
    work[i] + study[i],
    gb.GRB.LESS_EQUAL,
    1,
    name="Choice definition at time {0}".format(i)
    ) for i in range(n)}

# Optimization
model.optimize()

for v in model.getVars():
    print('%s %g' % (v.VarName, v.X))

print('Objective function: %g' % model.ObjVal)

work = [work[i].x for i in range(n)]
study = [study[i].x for i in range(n)]
money = [money[i].x for i in range(n)]
education = [education[i].x for i in range(n)]

print(work)
print(study)
print(money)
print(education)

# Time (x-axis)
time = list(range(n))

# Create the plot
plt.plot(time, work, label="Work")
plt.plot(time, study, label="Study")
plt.plot(time, money, label="Money")
plt.plot(time, education, label="Education")

# Display the plot
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

### Comments ###
# Non linearity constraint in the constraint of the transition function