import gurobipy as grb
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def transition(state, action):
    education_rate = 2

    if action == 'work':
        next_state = state  
    elif action == 'study':
        next_state = state+ education_rate

    # Clip the new state to ensure it doesn't go out of bounds
    next_state = min(max(next_state, 0), 10)
    return next_state

def Reward(state, action):
    base_salary = 1
    expenses = 1
    if action == "work":
        return -expenses + base_salary * (1 + state / 2)
    elif action == "study":
        return -expenses
    
def policy_dynamic_programing():
    number_of_periods = 20
    states = np.arange(0, 11, 1)  # Possible states I can be in in the last stage
    T = range(0,number_of_periods)

    action_space = ['work', 'study']  # Actions: work or study
    gamma = 0.95  # Discount factor 
    policy = np.empty((len(states),number_of_periods), dtype=object)  # Policy for each time step and state
    V = np.zeros((len(states),number_of_periods + 1))  # Value for each time step and state
    #Value of t=T
    first=True
    for t in reversed(T):
        values = []
        actions_taken=[]
        for state in states: 
            for action in action_space:   
                next_state = transition(state, action)
                reward = Reward(state,action)
                if first is True:
                    future_value=0
                else:
                    future_value = gamma*V[next_state, t+1]
                values.append(reward+future_value)
                actions_taken.append(action)

            V[state,t]= max(values)
            policy[state,t] = actions_taken[values.index(max(values))]    
        first=False

    
    return V, policy

V, policy =policy_dynamic_programing()
print(V,policy)


# Plotting
# Convert the 'work' and 'study' to numeric values (0 for 'work', 1 for 'study')
numeric_policy = np.where(policy == 'work', 0, 1)

# Create a color map (blue for 'work', green for 'study')
cmap = ListedColormap(['blue', 'green'])

# Plot the policy array as an image
plt.imshow(numeric_policy, cmap=cmap, aspect='auto')
# Invert the y-axis to display it in reverse order
plt.gca().invert_yaxis()

# Add labels, a color bar, and title
plt.colorbar(ticks=[0, 1], label='Policy')
plt.title('Policy: Work (Blue) vs Study (Green)')
plt.xlabel('Periods')
plt.ylabel('States')

# Display the plot
plt.show()