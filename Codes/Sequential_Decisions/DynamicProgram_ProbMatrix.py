import gurobipy as grb
from gurobipy import GRB
import numpy as np

# Transition function
def transition(state, action):
    education_rate = 2
    possible_next_states=[]

    if action == 'work':
        possible_next_states=[state]  
    elif action == 'study':
        possible_next_states=[state, state+ education_rate] 

    # Clip the new state to ensure it doesn't go out of bounds
    for i, next_state in enumerate(possible_next_states):
        possible_next_states[i] = min(max(next_state, 0), 10)
    return possible_next_states

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

    # Compute the probability matrix
    probability_matrix = np.zeros((len(states),len(states), len(action_space)))
    for state1 in states: 
        for state2 in states:
            for action_index, action in enumerate(action_space):
                if action=="work":
                    if state1 == state2:
                        probability_matrix[state1, state2, action_index] = 1
                if action=="study":
                    if state2 == state1+2:
                        probability_matrix[state1, state2, action_index] = 0.7
                    elif state2 == state1:
                        probability_matrix[state1, state2, action_index] = 0.3

                                  
    #Value of t=T
    first=True
    for t in reversed(T):
        values = []
        actions_taken=[]
        for state in states: 
            for action_index,action in enumerate(action_space): 
                prob_sum = 0  
                possible_next_states = transition(state, action) 
                reward = Reward(state,action)
                if first is True:
                    future_value=0
                else:
                    for next_state in possible_next_states: # Could also loop through all states as the next states 
                        #instead of using the transition function and since prob is 0, no value will be added
                        # e.g: for next_state in in states:   (and then deleting transition function)
                        prob_sum += probability_matrix[state, next_state, action_index]*V[next_state, t+1]
                    future_value = gamma * prob_sum
                values.append(reward+future_value)
                actions_taken.append(action)

            V[state,t]= max(values)
            policy[state,t] = actions_taken[values.index(max(values))]    
        first=False

    
    return V, policy
#V, policy =policy_dynamic_programing()
#print(V,policy)