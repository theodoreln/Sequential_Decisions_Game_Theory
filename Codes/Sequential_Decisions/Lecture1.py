import gurobipy as grb
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt
import random

def optimization(): #(Doesn't work because transition function is non-linear)
    # Define model
    model = grb.Model()

    # Define set
    number_of_periods = 20
    T = range(0,number_of_periods)


    # Define decision variables
    money_threshold = model.addVar(1, vtype=GRB.CONTINUOUS) #policy bound
    if_cond = model.addVars(number_of_periods, vtype=GRB.BINARY, name="policy_if")
    #work = model.addVars(number_of_periods, vtype= GRB.BINARY, name = "work")  
    #study = model.addVars(number_of_periods, vtype= GRB.BINARY, name = "study")  

    # Defne state variables
    money = model.addVars(number_of_periods, vtype= GRB.CONTINUOUS, name = "money") 
    education = model.addVars(number_of_periods, vtype= GRB.CONTINUOUS, name = "education") 

    # Decide if to work or to study sepending on the threshold
    work = model.addVars(number_of_periods, vtype= GRB.BINARY, name = "work")  
    study = model.addVars(number_of_periods, vtype= GRB.BINARY, name = "study") 

    # Parameters
    base_salary = 1
    education_rate = 2
    expenses = 1
    M = 1000 
    eps = 0.01


    # Bounds for theta work
    model.addConstr(money_threshold >=0)
    model.addConstr(money_threshold <=200)

    # Initial conditions 
    model.addConstr(money[0] == 3, name="initial_money")
    model.addConstr(education[0] == 0, name="initial_education")


    # Big M constraints
    model.addConstrs((money[t] >= money_threshold -M *(1-if_cond[t]) for t in T))
    model.addConstrs((money[t] <= money_threshold + M*if_cond[t] for t in T))

    model.addConstrs(((if_cond[t] == 1) >> (work[t] == 0) for t in T), name="indicator_constr")

    # Ensure that work and study are mutually exclusive (only one can be chosen at each period)
    model.addConstrs((work[t] + study[t] == 1 for t in T))

    # Update money and education over time
    model.addConstrs((money[t] == money[t-1] - expenses + work[t-1] * base_salary * (1 + education[t-1]/2) for t in range(1, number_of_periods)), name="money_update")
    model.addConstrs((education[t] == education[t-1] + study[t-1] * education_rate for t in range(1, number_of_periods)), name="education_update")

    # Define objective
    model.setObjective(grb.quicksum(money[t] for t in T), GRB.MAXIMIZE)

    model.optimize()


    for t in T:
        print(f"Period {t}: Work = {work[t].x}, Study = {study[t].x}, Money = {money[t].x}, Education = {education[t].x}")

    # Print the optimized theta values
    print(f"Optimized threshold: {money_threshold.x}")





# Transition function
def transition(state, action, education_rate):
    
    possible_next_states=[]

    if action == 'work':
        possible_next_states=[state]  
    elif action == 'study':
        possible_next_states=[state, state+ education_rate] 

    # Clip the new state to ensure it doesn't go out of bounds
    for i, next_state in enumerate(possible_next_states):
        possible_next_states[i] = min(max(next_state, 0), 10)
    return possible_next_states

# Reward function
def Reward(state, action):
    base_salary = 1
    expenses = 1
    if action == "work":
        return -expenses + base_salary * (1 + state / 2)
    elif action == "study":
        return -expenses


# Linear Program   (not finished)
def Linear_program():
    number_of_periods = 20
    states = np.arange(0, 11, 1)  # Possible states I can be in in the last stage
    T = range(0,number_of_periods)
    action_space = ['work', 'study']  # Actions: work or study
    gamma = 0.95  # Discount factor 
    education_rate = 2
    #policy = np.empty((len(states),number_of_periods), dtype=object)  # Policy for each time step and state
    #V = np.zeros((len(states),number_of_periods + 1))  # Value for each time step and state
   

    # Begin optimization model
    model = grb.Model()

    V = model.addVars(len(states),number_of_periods, vtype=GRB.CONTINUOUS, name="value")
    Q = model.addVars(len(states),number_of_periods,len(action_space), vtype=GRB.CONTINUOUS)


    # Constraints 
    model.addConstrs(V[state, t]>= Reward(state, action)+gamma*Reward(transition(state, action,education_rate),action)  #Reward(transition(state, action),action) = Value function of the next state (alternative define probability function of reaching next step)
                     for state in states for action in action_space for t in range(0,number_of_periods-1))
    model.addConstrs(V[state,number_of_periods-1]>= Reward(state, action) for state in states for action in action_space) # No future value at last time step 

    model.addConstrs(Q[state, t, index]== Reward(state, action)+gamma*Reward(transition(state, action, education_rate),action) 
                     for state in states for index,action in enumerate(action_space) for t in range(0,number_of_periods-1))
    model.addConstrs(Q[state,number_of_periods-1, index]== Reward(state, action) for state in states for index, action in enumerate(action_space)) # No future value at last time step 



    model.setObjective((grb.quicksum(grb.quicksum(V[state, t] for state in states) for t in T)), GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("Optimal values for V:")
        for state in range(len(states)):
            for t in range(number_of_periods):
                print(f"V[{state}, {t}] = {V[state, t].X}")

#Linear_program()

def compute_prob_matrix(number_of_periods, state_space, action_space, education_rate):
    probability_matrix = np.zeros((len(state_space),len(state_space), len(action_space)))
    for state1 in state_space: 
        for state2 in state_space:
            for action_index, action in enumerate(action_space):
                if action=="work":
                    if state1 == state2:
                        probability_matrix[state1, state2, action_index] = 1
                if action=="study":
                    if state2 == state1+education_rate:
                        probability_matrix[state1, state2, action_index] = 0.7
                    elif state2 == state1:
                        probability_matrix[state1, state2, action_index] = 0.3
    return probability_matrix


def policy_dynamic_programing():
    
    number_of_periods = 20
    states = np.arange(0, 11, 1)  # Possible states I can be in in the last stage
    T = range(0,number_of_periods)
    education_rate = 2
    action_space = ['work', 'study']  # Actions: work or study
    gamma = 0.95  # Discount factor 
    policy = np.empty((len(states),number_of_periods), dtype=object)  # Policy for each time step and state
    V = np.zeros((len(states),number_of_periods + 1))  # Value for each time step and state

    # Compute the probability matrix
    probability_matrix= compute_prob_matrix(number_of_periods,states,action_space, education_rate)
                                  
    #Value of t=T
    first=True
    for t in reversed(T):
        values = []
        actions_taken=[]
        for state in states: 
            for action_index,action in enumerate(action_space): 
                prob_sum = 0  
                possible_next_states = transition(state, action,education_rate) 
                reward = Reward(state,action)
                if first is True:
                    future_value=0
                else:
                    for next_state in possible_next_states: # Could also loop through all states as the next states 
                        #instead of using the transition function and since prob is 0, no value will be added
                        # e.g: for next_state in states:   (and then deleting transition function)
                        prob_sum += probability_matrix[state, next_state, action_index]*V[next_state, t+1]
                    future_value = gamma * prob_sum
                values.append(reward+future_value)
                actions_taken.append(action)

            V[state,t]= max(values)
            policy[state,t] = actions_taken[values.index(max(values))]    
        first=False

    
    return V, policy

# V, policy =policy_dynamic_programing()
# print(policy)



def valueIter(maxIter = 1000):
    number_of_periods = 20 
    T = range(number_of_periods)
    states = np.arange(0, 11, 1)  
    action_space = ['work', 'study'] 
    education_rate=2 
    gamma = 0.95  
    threshold = 1e-6  # Threshold for convergence
    policy = np.empty((len(states),number_of_periods), dtype=object)  # Policy for each time step and state
    V = np.zeros((len(states),number_of_periods)) 
    # Compute the probability matrix
    probability_matrix= compute_prob_matrix(number_of_periods,states,action_space,education_rate)

    # Value iteration
    iter = 0
    while True:
        iter+=1
        delta = 0  # Difference between old and new value functions
        new_V = np.copy(V)
        for t in T:
            for state in states:
                action_values = []
                for action_index, action in enumerate(action_space):
                    reward = Reward(state, action)
                    future_value = 0
                    if t < len(T)-1:
                        for next_state in states:
                            future_value += probability_matrix[state, next_state, action_index] * V[next_state, t+1]
                    action_value = reward + gamma * future_value
                    action_values.append(action_value)

                # Value function for current state
                new_V[state,t] = max(action_values)

                # Update the policy for the current state based on the action with the highest value
                policy[state,t] = action_space[np.argmax(action_values)]

                # Calculate the maximum difference for convergence check
                delta = max(delta, np.abs(new_V[state,t] - V[state,t]))

        # Update value function
        V = new_V

        # Check if the value function has converged
        if delta < threshold or iter>maxIter:
            break
    return V, policy

# V, policy =valueIter()
# print(policy)


def PolicyIter(maxIter = 1000):
    number_of_periods = 20 
    T = range(number_of_periods)
    states = np.arange(0, 11, 1)  
    action_space = ['work', 'study'] 
    education_rate=2 
    gamma = 0.95  
    threshold = 1e-6  # Threshold for convergence
    policy = np.random.choice(action_space, size=(len(states), number_of_periods))  # Policy for each time step and state
    V = np.zeros((len(states),number_of_periods)) 
    # Compute the probability matrix
    probability_matrix= compute_prob_matrix(number_of_periods,states,action_space,education_rate)

    iter = 0
    stable_policy = False

    while not stable_policy:
        iter += 1
        
        # Step 1: Policy Evaluation
        while True:
            delta = 0  # Difference between old and new value functions
            new_V = np.copy(V)
            
            for t in T:
                for state in states:    
                    action = policy[state, t]  # Take action according to the current policy
                    action_index = action_space.index(action)
                    
                    reward = Reward(state, action)  # Get the immediate reward
                    future_value = 0
                    if t < len(T)-1 :  # Compute the expected future value
                        for next_state in states:
                            future_value += probability_matrix[state, next_state, action_index] * V[next_state, t + 1]
                    
                    # Update value function for the current state
                    new_V[state, t] = reward + gamma * future_value

                    # Calculate the maximum difference for convergence check
                    delta = max(delta, np.abs(new_V[state, t] - V[state, t]))

            # Update the value function
            V = new_V

            # Break if the value function has converged
            if delta < threshold:
                break

        # Step 2: Policy Improvement
        stable_policy = True  # Assume the policy is stable unless we find a better action

        for t in T:
            for state in states:
                old_action = policy[state, t]  # Current action in the policy
                action_values = []
                
                # Evaluate each possible action
                for action_index, action in enumerate(action_space):
                    reward = Reward(state, action)
                    future_value = 0
                    if t < len(T)-1 :
                        for next_state in states:
                            future_value += probability_matrix[state, next_state, action_index] * V[next_state, t + 1]
                    action_value = reward + gamma * future_value
                    action_values.append(action_value)

                # Select the best action
                best_action = action_space[np.argmax(action_values)]
                
                # Update the policy if a better action is found
                if best_action != old_action:
                    stable_policy = False  # Policy is not stable, we need another iteration
                    policy[state, t] = best_action  # Update the policy with the best action

        # Stop if the policy is stable or if we exceed the maximum number of iterations
        if stable_policy or iter >= maxIter:
            break

    return V, policy

# V, policy =PolicyIter()
# print(policy)


def apply_policy(education_level, period, policy):

    specific_policy = policy[round(education_level), period]

    if specific_policy == "work":
        work = 1
        study = 0
    else:
        work = 0
        study = 1
    return work, study




def plot_appliedpolicy(policy):
    number_of_periods= 20
    T = range(0, number_of_periods)

    money = np.zeros(number_of_periods)
    education = np.zeros(number_of_periods)
    work = np.zeros(number_of_periods)
    study = np.zeros(number_of_periods)

    money[0] = 3
    education[0] = 0

    for period in range(1, number_of_periods):
        #expenses = random.choice([1,2])
        expenses = 1
        #base_salary = random.choice([2,3])
        base_salary = 1
        #education_rate = random.choice([2,3])
        education_rate = 2
        
        money[period] = money[period - 1] - expenses + work[period - 1]*(base_salary)*(1+ education[period - 1]/2)
        education[period] = education[period - 1] + study[period-1]*(education_rate)
        work[period], study[period] = apply_policy(education[period], (period-1), policy)
        if (work[period] + study[period] > 1) or (work[period] != 0 and work[period] != 1) or (study[period] != 0 and study[period] != 1):
            work[period], study[period] = 0, 0
            

    plt.plot(T, work, label='work', marker='o')
    plt.plot(T, study, label='study', marker='s')
    plt.plot(T, money, label='money', marker='^')
    plt.plot(T, education, label='education', marker='x')

    # Add labels and title
    plt.xlabel('time')
    #plt.ylabel('Y-axis label')
    #plt.title('Plot of Four 5-Element Arrays')
    plt.legend()  # Show the legend

    # Show the plot
    plt.show()

VD, PolicyD = policy_dynamic_programing()
VV, PolicyV = valueIter()
VP, policyP = PolicyIter()

# plot_appliedpolicy(PolicyD)
# plot_appliedpolicy(PolicyV)
# plot_appliedpolicy(policyP)