import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import GRB

# Define constants
numPlayers = 10
base_salary = 10
education_cost_coefficient = 15
gamma = 0.95

# Possible education from 0 to 1 with 0.1 increments
education_values = np.linspace(0, 1, 11)

# Definition of base functions
# The goal is to construct a learning process with vanishing regret

def calculate_average_education(education_levels):
    return np.mean(education_levels)

def salary(player_education, avg_education):
    return base_salary * (1 + player_education / avg_education)

def cost_of_education(education_level):
    return education_cost_coefficient * education_level

def money(player_education, avg_education):
    if avg_education == 0:
        return base_salary - cost_of_education(player_education)
    else:
        return salary(player_education, avg_education) - cost_of_education(player_education)

def regret(player_id, best_pure_strategy, historical_strategies, historical_payoffs):
    # Compute the discounted average payoff for the best pure strategy
    discounted_sum_best_pure_strategy = 0
    discount_factor = 1
    for t in range(historical_strategies.shape[0] - 1, -1, -1):
        historical_strategy = historical_strategies[t, :].copy()
        historical_strategy[player_id] = best_pure_strategy
        discounted_sum_best_pure_strategy += discount_factor * money(best_pure_strategy, calculate_average_education(historical_strategy))
        discount_factor *= gamma
    avg_payoff_best_pure_strategy = discounted_sum_best_pure_strategy / historical_strategies.shape[0]
    
    # Compute the discounted average payoff for the historical strategies
    discounted_sum_historical = 0
    discount_factor = 1
    for t in range(historical_payoffs.shape[0] - 1, -1, -1):
        discounted_sum_historical += discount_factor * historical_payoffs[t, player_id]
        discount_factor *= gamma
    avg_payoff_historical = discounted_sum_historical / historical_payoffs.shape[0]
    
    return avg_payoff_best_pure_strategy - avg_payoff_historical

def calculate_highest_regret(player_id, historical_strategies, historical_payoffs):
    regrets = []
    for e in education_values:
        r = regret(player_id, e, historical_strategies, historical_payoffs)
        regrets.append(r)
    regrets = np.array(regrets)
    
    # Set negative regrets to 0
    regrets[regrets < 0] = 0
    
    # Calculate probabilities proportional to the regrets
    total_regret = np.sum(regrets)
    if total_regret == 0:
        probabilities = np.ones_like(regrets) / len(regrets)
    else:
        probabilities = regrets / total_regret
    
    # Choose an action based on the calculated probabilities
    chosen_index = np.random.choice(len(education_values), p=probabilities)
    best_pure_strategy = education_values[chosen_index]
    highest_regret = regrets[chosen_index]
    
    return best_pure_strategy, highest_regret

def fictitious_play(player_id, historical_strategies, historical_payoffs):
    # Compute the most choosen strategies (discounted) of other players in the last 10 rounds
    most_chosen_strategies = np.zeros(10)
    for t in range(historical_strategies.shape[0] - 1, -1, -1):
        most_chosen_strategies += historical_strategies[t, :]
    most_chosen_strategies /= historical_strategies.shape[0]
    most_chosen_strategies = np.round(most_chosen_strategies * 10) / 10
    
    # Compute the best response to the most chosen strategies of other players
    best_response = 0
    best_payoff = -np.inf
    for e in education_values:
        strategy = most_chosen_strategies.copy()
        strategy[player_id] = e
        payoff = money(e, calculate_average_education(strategy))
        if payoff > best_payoff:
            best_payoff = payoff
            best_response = e
            
    # Calculate discounted regret with the best response
    discounted_sum_best_response = 0
    discount_factor = 1
    for t in range(historical_strategies.shape[0] - 1, -1, -1):
        historical_strategy = historical_strategies[t, :].copy()
        historical_strategy[player_id] = best_response
        discounted_sum_best_response += discount_factor * money(best_response, calculate_average_education(historical_strategy))
        discount_factor *= gamma
    avg_payoff_best_response = discounted_sum_best_response / historical_strategies.shape[0]
    
    # Compute the discounted average payoff for the historical strategies
    discounted_sum_historical = 0
    discount_factor = 1
    for t in range(historical_payoffs.shape[0] - 1, -1, -1):
        discounted_sum_historical += discount_factor * historical_payoffs[t, player_id]
        discount_factor *= gamma
    avg_payoff_historical = discounted_sum_historical / historical_payoffs.shape[0]
    
    return best_response, avg_payoff_best_response - avg_payoff_historical

def weight_function(player_id, historical_strategies, historical_payoffs, historical_weights):
    # Action is chosen based on probabilities proportional to the weights
    probabilities = historical_weights[-1, player_id, :] / np.sum(historical_weights[-1, player_id, :])
    chosen_index = np.random.choice(len(education_values), p=probabilities)
    best_pure_strategy = education_values[chosen_index]
    
    # Compute the regret of the chosen action
    r = regret(player_id, best_pure_strategy, historical_strategies, historical_payoffs)
    
    return best_pure_strategy, r
    

def learning_process(numPlayers, numIterations, type):
    # Initialize the learning process
    historical_strategies = np.zeros((numIterations, numPlayers))
    historical_payoffs = np.zeros((numIterations, numPlayers))
    historical_regrets = np.zeros((numIterations, numPlayers))
    historical_weights = np.zeros((numIterations, numPlayers, len(education_values)))
    
    for t in range(numIterations):
        print("Iteration", t)
        for i in range(numPlayers):
            if t < 10 :
                historical_strategies[t, i] = constant_strategies[i]
                historical_regrets[t, i] = 0  # No regret in the first iterations
                historical_weights[t, i] = 1  # Equal weights in the first iterations
            else:
                if type == "regret_matching":
                    best_strat, highest_reg= calculate_highest_regret(i, historical_strategies[:t, :], historical_payoffs[:t, :])
                if type == "fictitious_play":
                    best_strat, highest_reg= fictitious_play(i, historical_strategies[:t, :], historical_payoffs[:t, :])
                if type == "multiplicative_weights":
                    best_strat, highest_reg= weight_function(i, historical_strategies[:t, :], historical_payoffs[:t, :], historical_weights[:t, :, :])
                historical_strategies[t, i] = best_strat
                historical_regrets[t, i] = highest_reg
        for i in range(numPlayers):
            historical_payoffs[t, i] = money(historical_strategies[t, i], calculate_average_education(historical_strategies[t, :]))
            if type == "multiplicative_weights" and t >= 10 :
                historical_weights[t, i, :] = historical_weights[t-1, i, :] + np.sqrt(historical_payoffs[t, i])
    
    return historical_strategies, historical_regrets

# Run the learning process
numIterations = 75
constant_strategies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
historical_strategies, historical_regrets = learning_process(numPlayers, numIterations, "regret_matching")
print(historical_strategies)
print(historical_regrets)

# Plot the learning process
plt.figure()
for i in range(numPlayers):
    plt.plot(historical_strategies[:, i], label="Player " + str(i+1))
plt.xlabel("Iteration")
plt.ylabel("Education Level")
plt.legend()
plt.show()

# Plot the regret
plt.figure()
for i in range(numPlayers):
    plt.plot(historical_regrets[:, i], label="Player " + str(i+1))
plt.xlabel("Iteration")
plt.ylabel("Regret")
plt.legend()
plt.show()

# 75 for regret matching, 200 for fictitious play, 350 for multiplicative weights