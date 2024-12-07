############# Policy iteration ##############

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Function to take the best decision
def Policy_Policy_iteration(n,base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba) :
    # Creating array Best_dec with random values
    Best_dec = np.random.randint(2, size=(n,max_education+1-base_education))
    V_array = np.zeros((n,max_education+1-base_education))

    # Recursion to repeat computation of V
    cond = 0
    while cond != 1 :
        cond_sum = 0
        for i in range(n-1, -1, -1) :
            for j in range(max_education - base_education, -1, -1) :
                if i == n-1 :
                    V_array[i,j] = ret(j+base_education,Best_dec[i,j])
                else :
                    V_array[i,j] = ret(j+base_education,Best_dec[i,j]) + gamma * sum([proba(l,j,Best_dec[i,j])*V_array[i+1,l] for l in range(j,max_education-base_education+1)])
        Best_new = np.zeros((n,max_education+1-base_education))
        for i in range(n-1, -1, -1) :
            for j in range(max_education - base_education, -1, -1) :
                if i == n-1 :
                    if ret(j+base_education,0) > ret(j+base_education,1) :
                        Best_new[i,j] = 0
                    else :
                        Best_new[i,j] = 1
                else :
                    if (ret(j+base_education,0) + gamma * sum([proba(l,j,0)*V_array[i+1,l] for l in range(j,max_education-base_education+1)])) > (ret(j+base_education,1) + gamma * sum([proba(l,j,1)*V_array[i+1,l] for l in range(j,max_education-base_education+1)]) ) :
                        Best_new[i,j] = 0
                    else :
                        Best_new[i,j] = 1
                cond_sum += (Best_dec[i,j] - Best_new[i,j])
        Best_dec = Best_new.copy()
        if cond_sum == 0 :
            cond = 1
    
    return(Best_dec)



if __name__ == "__main__":
    ### Parameters 
    # Numbers of periods
    n = 10
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
        plt.yticks(np.arange(data.shape[0]), np.arange(base_education, base_education + data.shape[0]))
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
        
    Best_dec = Policy_Policy_iteration(n,base_salary, base_education, expenses, education_rate, max_education, gamma, ret, proba)
    Plotting(Best_dec)
