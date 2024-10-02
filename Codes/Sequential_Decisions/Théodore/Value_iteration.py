############# Value iteration ##############

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Function to take the best decision
def Policy_Value_iteration(n,base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba) :
    # Creating array V with random values
    V_array = np.random.uniform(0, ret(max_education, 0), (n,max_education+1-base_education))

    # Recursion to repeat computation of V
    cond = 0
    while cond != 1 :
        V_new = np.zeros((n,max_education+1-base_education))
        cond_sum = 0
        for i in range(n-1, -1, -1) :
            for j in range(max_education, -1, -1) :
                if i == n-1 :
                    V_new[i,j] = max([ret(j,0), ret(j,1)])
                else :
                    V_new[i,j] = max([ret(j,0) + gamma * sum([proba(l,j,0)*V_new[i+1,l] for l in range(j,min(max_education,j+2)+1)]), 
                                    ret(j,1) + gamma * sum([proba(l,j,1)*V_new[i+1,l] for l in range(j,min(max_education,j+3)+1)])])
                cond_sum += (V_array[i,j] - V_new[i,j])
        V_array = V_new.copy()
        if cond_sum == 0 :
            cond = 1

    # Creating array Q and recursion to fill it
    Q_array = np.zeros((n,max_education+1-base_education,2))
    for i in range(n-1, -1, -1) :
            for j in range(max_education, -1, -1) :
                if i == n-1 :
                    Q_array[i,j,0] = ret(j,0)
                    Q_array[i,j,1] = ret(j,1)
                else :
                    Q_array[i,j,0] = ret(j,0) + gamma * sum([proba(l,j,0)*V_array[i+1,l] for l in range(j,min(max_education,j+2)+1)])
                    Q_array[i,j,1] = ret(j,1) + gamma * sum([proba(l,j,1)*V_array[i+1,l] for l in range(j,min(max_education,j+2)+1)])
                    
    # Best_dec array
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
        
    Best_dec = Policy_Value_iteration(n,base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba)
    Plotting(Best_dec)
