############# Value function linear programming ##############

import gurobipy as gb
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Function to take the best decision
def Policy_LP(n,base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba) :
    # Define the model
    model = gb.Model("work_study_model") 

    # Set time limit
    model.Params.TimeLimit = 100

    # Add variables 
    V = {(i,j): model.addVar(vtype = GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Value function at time {0} and education level {1}".format(i,j)) 
        for i in range(n) for j in range(base_education,max_education+1)}
    Q = {(i,j,k): model.addVar(vtype = GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q function at time {0}, education level {1} and decision {2}".format(i,j,k)) 
        for i in range(n) for j in range(base_education,max_education+1) for k in range(2)}

    # Set objective function
    objective = gb.quicksum(V[i,j] for i in range(n) for j in range(base_education, max_education+1))
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    V_constraint = {(i,j,k) : model.addConstr(
        V[i,j],
        gb.GRB.GREATER_EQUAL,
        ret(j, k) + gamma * gb.quicksum(proba(l, j, k) * V[i+1, l] for l in range(j, max_education+1)),
        name="Value function definition at time {0}, education level {1} and decision {2}".format(i,j,k)
        ) for i in range(n-1) for j in range(base_education, max_education+1) for k in range(2)}
    for j in range(base_education, max_education+1) :
        for k in range(2) :
            V_constraint[n-1,j,k] = model.addConstr(V[n-1,j], gb.GRB.GREATER_EQUAL, ret(j, k), 
                                                name="Value function definition at time {0}, education level {1} and decision {2}".format(n-1,j,k))

    Q_constraint = {(i,j,k) : model.addConstr(
        Q[i,j,k],
        gb.GRB.EQUAL,
        ret(j, k) + gamma * gb.quicksum(proba(l, j, k) * V[i+1, l] for l in range(j, max_education+1)),
        name="Q function definition at time {0}, education level {1} and decision {2}".format(i,j,k)
        ) for i in range(n-1) for j in range(base_education, max_education+1) for k in range(2)}
    for j in range(base_education, max_education+1) :
        for k in range(2) :
            Q_constraint[n-1,j,k] = model.addConstr(Q[n-1,j,k], gb.GRB.EQUAL, ret(j, k), 
                                                name="Q function definition at time {0}, education level {1} and decision {2}".format(n-1,j,k))

    # Optimization
    model.optimize()

    # for v in model.getVars():
    #     print('%s %g' % (v.VarName, v.X))

    # print('Objective function: %g' % model.ObjVal)


    V_array = np.array([[V[i,j].x for j in range(base_education, max_education+1)] for i in range(n)])
    Q_array = np.array([[[Q[i,j,k].x for k in range(2)] for j in range(base_education, max_education+1)] for i in range(n)])

    Best_dec = np.argmax(Q_array, axis=2)
    
    return(Best_dec)


if __name__ == "__main__":
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
        plt.yticks(np.arange(data.shape[0]), np.arange(1, data.shape[0]+1))
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
        
    Best_dec = Policy_LP(n,base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba)
    Plotting(Best_dec)
