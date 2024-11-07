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

    if action == 'work' or action == 0:
        possible_next_states=[state]  
    elif action == 'study' or action == 1:
        possible_next_states=[state, state+ education_rate] 

    # Clip the new state to ensure it doesn't go out of bounds
    for i, next_state in enumerate(possible_next_states):
        possible_next_states[i] = min(max(next_state, 0), 10)

    return possible_next_states

# Reward function
def Reward(state, action):
    base_salary = 1
    expenses = 1
    if action == "work" or action ==0:
        return -expenses + base_salary * (1 + state / 2)
    elif action == "study" or action ==1:
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

def compute_prob_matrix(number_of_periods, state_space, action_space, education_rate): # number of periods not needed, delete!
    probability_matrix = np.zeros((len(state_space)+education_rate,len(state_space)+education_rate, len(action_space)))
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


def apply_policy(education_level, period, policy, money):

    specific_policy = policy[round(education_level), period]
    expenses = 1

    if specific_policy == "study":
        if money>=expenses:  # To ensure to not get beneath 0
            work = 0
            study = 1
        else:
            work=1
            study=0
    else:
        work = 1
        study = 0
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
        work[period], study[period] = apply_policy(education[period], (period-1), policy, money[period])
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

#VD, PolicyD = policy_dynamic_programing()
#VV, PolicyV = valueIter()
# VP, policyP = PolicyIter()

# plot_appliedpolicy(PolicyD)
# plot_appliedpolicy(PolicyV)
# plot_appliedpolicy(policyP)


def policy_approximation(state_divider, stage_divider):
    '''The state_divider specifies how often every state is used in the subset. For example, if state_divider=3, 
    every third state in the state_subset will be reused. 
    The stage_divider can either be a tuple or an integer:
        -If it’s an integer, it works similarly to state_divider, meaning it divides or skips through stages in the same way.
        -If it’s a tuple, it indicates the start and end indices of the entire time set, so that the time_subset will consist 
        of the time set starting from the Start-Index and ending at the End-Index'''
    states = np.arange(0, 10, 1)
    number_of_periods = 20
    T=range(number_of_periods)
    known_policy_words = PolicyIter()[1]
    known_policy_num = [[1 if s == "study" else 0 for s in inner_list] for inner_list in known_policy_words]
    states_sub = states[0::state_divider]

    if isinstance(stage_divider, tuple):
        subset_t_start = stage_divider[0]
        subset_t_end = stage_divider[1]
        T_sub = T[subset_t_start:subset_t_end]
        policy_subset = [row[subset_t_start:subset_t_end] for row in known_policy_num[0::state_divider]]#known_policy_num[0::state_divider, subset_t_start:subset_t_end]
    else:
        T_sub = T[0::stage_divider]
        policy_subset = [row[0::stage_divider] for row in known_policy_num[0::state_divider]]#known_policy_num[0::state_divider, 0::stage_divider]

    # Create model
    model = grb.Model()
    # Add decision variables to model
    alpha_value = model.addVar(vtype=GRB.CONTINUOUS, name="alpha")
    beta_value = model.addVar(vtype=GRB.CONTINUOUS, name = "beta")
    gamma_value = model.addVar(vtype=GRB.CONTINUOUS, name = "gamma")

    # Constraints (Set in objective function)
    # Objective
    model.setObjective((grb.quicksum(((alpha_value- beta_value* state - gamma_value*t)- policy_subset[Sindex][Tindex])**2 for Tindex, t in enumerate(T_sub) for Sindex, state in enumerate(states_sub))), GRB.MINIMIZE) #for Tindex in range(len(T_sub)) for Sindex in range(len(states_sub))
    model.Params.OutputFlag = 0
    model.optimize()
   
    if model.status == GRB.OPTIMAL:
        # print("Optimal values for alpha, beta and gamma:")
        # print(f"alpha = {alpha_value.X}, beta = {beta_value.X}, gamma = {gamma_value.X}")
        # print("The policy function is:")
        # print(f"study =  {alpha_value.X}- {beta_value.X} * education_level - {gamma_value.X} * stage")

        # Test performance on whole policy set
        error = sum([(((alpha_value.X- beta_value.X* state - gamma_value.X*t) - known_policy_num[state][t])**2) for state in states for t in T])
        print(f"Least squared error: {error}, Policy function: study = {alpha_value.X}- {beta_value.X} * education_level - {gamma_value.X} * stage")
        return(error, alpha_value.X, beta_value.X, gamma_value.X)


# for i in [2,3,4]:
#     for x in [2,3,4]:
#         policy_approximation(i, x) #best working (smallest error) for [4,2], so for every fourth state and every second time step (Shouldn't it be [2,2]?)

# policy_approximation(2,(0,7))
# policy_approximation(2,(7,14)) # best working of the onesthat have a whole batch of times 
# policy_approximation(2,(14,20))
# policy_approximation(2,(17,20))
# policy_approximation(2,(10,13))
# policy_approximation(2,(0,3))




def value_approximation():
    # V(E,t) = alpha*E + beta * t + gamma * E*t
    states = np.arange(0, 10, 1)
    number_of_periods = 20
    action_space= ["work","study"]
    N_rollouts = 10000
    discount = 0.95
    education_rate = 2
    probability_matrix= compute_prob_matrix(number_of_periods,states,action_space,education_rate)
    learning_rate = 1e-10 
    
    # 1. Come up with a functional form for the Value Function: V(E,t) = alpha*E + beta * t + gamma * E*t
    # Initialization of parameters 
    alpha = 10
    beta = 10
    gamma = 10
    stud = 0
    wo = 0
    # 3. Update theta using gradient descent

    tolerance = 1e-6  # Convergence tolerance for the loss function
    previous_loss = float('inf')

    for epoch in range(N_rollouts):
        total_loss = 0  # Initialize total loss for this epoch
        #Set initial value for education level at t=0
        # E_t = random.randint(0, len(states)-1)
        # V_t = alpha * E_t
        #for t in range(number_of_periods-1):

        t = random.randint(0, number_of_periods)
        E_t = random.randint(0, len(states))
        V_t =  alpha * E_t + beta * t + gamma * E_t * t
        # next states value 
        E_next_work = E_t
        #E_next_study = E_t + education_rate *probability_matrix(E_t, E_t+education_rate, 1) #?
        next_states_study = transition(E_t, 1,education_rate)
        prob1 = probability_matrix[E_t, next_states_study[0], 1]
        prob2 = probability_matrix[E_t, next_states_study[1], 1]
        if prob2 != 0:
            E_next_study = random.choices(next_states_study, weights= (prob1*100,prob2*100), k=1)[0]
        else:
            E_next_study = E_t
        V_next_work = alpha * E_next_work + beta * (t + 1) + gamma * E_next_work * (t + 1)
        V_next_study = alpha * E_next_study + beta * (t + 1) + gamma * E_next_study * (t + 1)
        V_next = max(V_next_study,V_next_work)
        
        if V_next == V_next_study:
            stud+=1
            action = 1
            E_next = E_next_study
        elif V_next == V_next_work:
            wo+=1
            action = 0
            E_next = E_next_work    
        # Reward for this state taken action with higher next value function
        R_t = Reward(E_t, action)
        
        # Bellman error (temporal difference error)
        delta = (V_t- (R_t + discount * V_next))
        total_loss += delta**2
        # Gradient descent
        partial_alpha = 2 * delta * (E_t - discount * E_next)  #partial derivative: ddelta/dalpha =outer*inner 
        partial_beta = 2 * delta * (t - discount * (t + 1))
        partial_gamma = 2 * delta * (E_t * t - discount * E_next * (t + 1))
        # Update theta
        alpha -= learning_rate * partial_alpha
        beta -= learning_rate * partial_beta
        gamma -= learning_rate * partial_gamma


        # Calculate average loss for this epoch
        avg_loss = total_loss / (N_rollouts * (number_of_periods - 1))

        # Check convergence
        if abs(previous_loss - avg_loss) < tolerance:
            print(f"Converged at epoch {epoch} with stud={stud} and work = {wo}")
            break
        previous_loss = avg_loss


    print(f"V(E,t) = {alpha}*E + {beta} * t + {gamma} * E*t with stud={stud} and work = {wo}")
    return (alpha, beta, gamma)


alpha, beta, gamma = value_approximation()


def V(period, education):
    return alpha * education + beta * period + gamma * education * period

states = np.arange(0, 10, 1)
number_of_periods = 20
action_space= ["work","study"]
education_rate = 2
#%% Policy
probability_matrix= compute_prob_matrix(number_of_periods,states,action_space,education_rate)
def policy(education, period):
    # Let us compute Q for all the actions possibles (work or study)  
    # the action is work
    base_salary = 1
    expenses = 1
    discount = 0.95
    Q_w = - expenses + base_salary * (1 + education/2) + discount * V(period+1, education)
    # the action is study
    if education == 8 :
        education_rate=1
        Q_s = - expenses + discount* (probability_matrix[education, education+education_rate, 1] * V(period+1, education + education_rate)+probability_matrix[education, education, 1] * V(period+1, education))
    elif education == 9:
        Q_s =- expenses + discount * V(period+1, education)
    else:
        education_rate=2 
        Q_s = - expenses + discount* (probability_matrix[education, education+education_rate, 1] * V(period+1, education + education_rate)+probability_matrix[education, education, 1] * V(period+1, education))

    if Q_w > Q_s : # the action to take is work
        return 0
    else : # the action to take is study
        return 1
    
states = np.arange(0, 10, 1)
number_of_periods = 20
P = np.zeros((len(states),number_of_periods))
for education in range(len(states)):
    for t in range(number_of_periods):
        P[education, t] = policy(round(education), t)

print(P)

## best output so far with very low learning rate:
# [[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]]
# %%
