# VCG mechanism for a system with 4 players
import numpy as np
import gurobipy as gb
from gurobipy import GRB
import matplotlib.pyplot as plt

### Function to compute the VCG mechanism
# Cost of the player
def cost(i, x, alpha, beta):
    return alpha[i]*x + beta[i]*x**2

# Optimization of production levels for a given number of players
def optimal_production_levels(alpha_bid: list, beta_bid: list):
    # Number of players
    n = len(alpha_bid)
    
    # Create a new model
    model = gb.Model("Opt_Prod_Levels")
    
    # Create variables
    x = {i: model.addVar(vtype=GRB.CONTINUOUS, name=f"Production level of player {i}") for i in range(n)}
    
    # Set objective
    model.setObjective(gb.quicksum(cost(i, x[i], alpha_bid, beta_bid) for i in range(n)), GRB.MINIMIZE)
    
    # Add constraints
    model.addConstr(gb.quicksum(x[i] for i in range(n)) == xD)
    
    # Optimize
    model.optimize()
    
    # Return the optimal production levels
    return [x[i].x for i in range(n)]

# Compute the profit of player i with the VCG mechanism
def Profit(i, alpha, beta, alpha_bid, beta_bid):
    # Compute the optimal production levels when the player is here
    x = optimal_production_levels(alpha_bid, beta_bid)
    # Sum of the cost of the others players
    sum_other_players = sum([cost(j, x[j], alpha_bid, beta_bid) for j in range(len(alpha_bid)) if j != i])
    
    # Compute the optimal production levels when the player is not here
    alpha_bid_without_i = np.delete(alpha_bid, i)
    beta_bid_without_i = np.delete(beta_bid, i)
    x_without_i = optimal_production_levels(alpha_bid_without_i, beta_bid_without_i)
    # Sum of the cost of the others players
    sum_other_players_without_i = sum([cost(j, x_without_i[j], alpha_bid_without_i, beta_bid_without_i) for j in range(len(alpha_bid_without_i))])
    
    # Cost of the player i
    cost_i = cost(i, x[i], alpha, beta)
    
    # Return the profit of the player i
    return sum_other_players_without_i - sum_other_players - cost_i

# Iterate the profit calculation on the choice of bid for the player i, make alpha vary between 0 and 10 with step 0.5 and beta vary between -1 and -0.05 with step 0.05
def iterate_profit(i, alpha, beta, alpha_bid, beta_bid):
    # 2D array to store the profit of the player i
    profit = np.zeros((21, 20))
    # Iterate on the alpha and beta values
    for j in range(21):
        for k in range(20):
            alpha_bid[i] = j * 0.5
            beta_bid[i] =  0.05 + k * 0.05
            profit[j, k] = Profit(i, alpha, beta, alpha_bid, beta_bid)
    
    # Plot this profit array in a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(0, 10.5, 0.5), np.arange(0.05, 1.05, 0.05))
    X, Y = X.T, Y.T  # Transpose to match the shape of profit
    ax.plot_surface(X, Y, profit, cmap='viridis')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Profit')
    plt.show()
    
    # Return the profit array
    return profit


# When launching the script as main 
if __name__ == "__main__":
    ### Input parameters
    # Alpha costs of the players
    alpha = np.array([5, 4, 6, 2])
    # Beta costs of the players
    beta = np.array([0.3, 0.5, 0.3, 0.1])
    # Input alpha bids 
    alpha_bid = np.array([5, 4, 6, 2])
    # Input beta bids
    beta_bid = np.array([0.3, 0.5, 0.3, 0.1])
    # Demand of the market
    xD = 20
    # Player i
    i = 0
    
    # Compute the optimal production levels
    print(optimal_production_levels(alpha_bid, beta_bid))
    # Compute the profit of the player i
    print(Profit(i, alpha, beta, alpha_bid, beta_bid))
    # Show profit of player 0 with iterate_profit
    profit = iterate_profit(i, alpha, beta, alpha_bid, beta_bid)
    # Find argmax of the maximum in profit
    argmax = np.argmax(profit)
    # Find the alpha and beta values corresponding to the argmax
    alpha_sol = argmax // 20 * 0.5
    beta_sol = 0.05 + argmax % 20 * 0.05
    print(f"Player {i+1} - Optimal alpha: {alpha_sol}, optimal beta: {beta_sol} - Profit : {profit[argmax // 20, argmax % 20]}")