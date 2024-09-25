import numpy as np
import matplotlib.pyplot as plt

# Define constants
numPlayers = 10
base_salary = 1000
education_cost_coefficient = 150

# Initialize random education levels for all players
education_levels = np.round(np.random.uniform(0, 10, numPlayers), 1)

def calculate_average_education(education_levels):
    return np.mean(education_levels)

def salary(player_education, avg_education):
    return base_salary * (1 + player_education / avg_education)

def cost_of_education(education_level):
    return education_cost_coefficient * education_level

def money(player_education, avg_education):
    return salary(player_education, avg_education) - cost_of_education(player_education)

# Function to compute the best response for a player given others' choices
def best_response(player_index, education_levels):
    avg_education = calculate_average_education(education_levels)
    best_education = 0
    best_payoff = -np.inf
    
    # Try different levels of education to find the best response
    for e in np.linspace(0, 10, 101):  # Assuming education levels between 0 and 10
        temp_education_levels = education_levels.copy()
        temp_education_levels[player_index] = e
        avg_education = calculate_average_education(temp_education_levels)
        payoff = money(e, avg_education)
        if payoff > best_payoff:
            best_education = e
            best_payoff = payoff
    
    return best_education

# Best Response Dynamics loop
def best_response_dynamics(education_levels, iterations):
    history = np.zeros((iterations+1, numPlayers))
    history[0, :] = education_levels  # Store initial values
    
    for t in range(iterations):
        for player in range(numPlayers):
            education_levels[player] = best_response(player, education_levels)
        history[t+1, :] = education_levels  # Store education levels at each iteration
        print(f"Average Education after iteration: {calculate_average_education(education_levels)}")
    return history

# Run the dynamics
iterations = 100
history = best_response_dynamics(education_levels, iterations)
print("Final education levels:", history[iterations,:])

# Plot the education levels over iterations for each player
plt.figure(figsize=(10, 6))
for player in range(numPlayers):
    plt.plot(range(iterations+1), history[:, player], label=f'Player {player+1}')

# Customize plot
plt.xlabel('Iterations')
plt.ylabel('Education Level')
plt.title('Education Levels Over Iterations for Each Player')
plt.legend(loc='best')
plt.grid(False)  # Disable the grid
plt.show()