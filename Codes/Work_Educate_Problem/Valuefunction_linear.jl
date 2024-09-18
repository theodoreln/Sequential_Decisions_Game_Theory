############## Value function linear programming ##############

using JuMP, Gurobi, Plots
gr()

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
# Max education
max_education = base_education + n * 2
# Discount factor
gamma = 0.95

# Probability function of arriving to stage x' with choosing action u in stage x
function proba(edu_new, edu_now, dec_now)
    # Difference between current and old education
    diff_edu = edu_new - edu_now
    # If we choose to work, we are sure to stay at the same education
    if dec_now == 0
        if diff_edu == 2
            return 0
        elseif diff_edu == 1
            return 0
        elseif diff_edu == 0
            return 1
        elseif diff_edu > 2 || diff_edu < 0
            return 0
        end
        # If we choose to study, we have a chance to improve by 2,1, or 0 our education level
    elseif dec_now == 1
        if diff_edu == 2
            return 0.5
        elseif diff_edu == 1
            return 0.4
        elseif diff_edu == 0
            return 0.1
        elseif diff_edu > 2 || diff_edu < 0
            return 0
        end
    end
end

# Return function at a stage x for an action u 
function ret(edu_now, dec_now)
    return ((1 - dec_now) * (base_salary * (1 + edu_now / 2)) - expenses)
end


# List of time
time = collect(1:n)'
# List of Education possibilities
education = collect(base_education:max_education)'
length_education = length(education)
# List of Decision possibilities, 0 is working and 1 is studying
decision = [0 1]

# Model optimization
model = Model(Gurobi.Optimizer)

@variable(model, V[1:n, 1:length_education])
@variable(model, Q[1:n, 1:length_education, 1:2])

@objective(model, Min, sum(sum(V[i, j] for i = 1:n) for j = 1:length_education))

# Constraint 
@constraint(model, [i = 1:n-1, j = 1:length_education, k = 1:2], V[i, j] >= ret(education[j], decision[k]) + gamma * sum(proba(education[l], education[j], decision[k]) * V[i+1, l] for l = j:length_education))
@constraint(model, [j = 1:length_education, k = 1:2], V[n, j] >= ret(education[j], decision[k]))
# Computation of Q
@constraint(model, [i = 1:n-1, j = 1:length_education, k = 1:2], Q[i, j, k] == ret(education[j], decision[k]) + gamma * sum(proba(education[l], education[j], decision[k]) * V[i+1, l] for l = j:length_education))
@constraint(model, [j = 1:length_education, k = 1:2], Q[n, j, k] == ret(education[j], decision[k]))

JuMP.optimize!(model)

Valuefunction = value.(V)
Qfunction = value.(Q)

# Vector with best decision for every state in every time step, -1 if not reachable, 0 if study, 1 if work
Best_Dec = zeros(n, length_education)
for i = 1:n
    for j = 1:length_education
        if Qfunction[i, j, 1] >= Qfunction[i, j, 2]
            Best_Dec[i, j] = 0
        else
            Best_Dec[i, j] = 1
        end
        real_education = education[j]
        if real_education > base_education + 2 * (i - 1)
            Best_Dec[i, j] = -1
        end
    end
end

# Giving the best decision for a given state
function best_decision(time_step, education_step)
    indice = argmax(Qfunction[time_step, education_step, :])
    real_education = education[education_step]
    if indice == 1
        println("You should work at time step $time_step and education $real_education")
        if real_education > base_education + 2 * (time_step - 1)
            println("It is not possible to have reach this education $real_education at time step $time_step")
        end
    elseif indice == 2
        println("You should educate yourself at time step $time_step and education $real_education")
        if real_education > base_education + 2 * (time_step - 1)
            println("It is not possible to have reach this education $real_education at time step $time_step")
        end
    end
end

# Example matrix with three possible values (-1, 0, 1)
matrix = Best_Dec

# Initialize an empty plot
p = plot(xlim=(0.5, n + 0.5), ylim=(base_education - 0.5, max_education + 0.5),
    ratio=:equal, legend=false, framestyle=:box, grid=true,
    xlabel="Time", ylabel="Education level",
    xticks=1:n, yticks=base_education:max_education,
    xtickfont=font(4), ytickfont=font(4), title="Work / Study Decision")

# Loop through the matrix and plot squares with appropriate colors and annotations
for i in 1:n
    for j in 1:length_education
        value = matrix[i, j]
        x = i  # Row -> x-axis
        y = education[j]  # Column -> y-axis

        # Define color and text based on the value
        if value == -1
            rect = Shape(x .+ [-0.5, 0.5, 0.5, -0.5], y .+ [-0.5, -0.5, 0.5, 0.5])
            plot!(rect, fillcolor=:grey, linecolor=:black, label="Irreachable")
        elseif value == 1
            rect = Shape(x .+ [-0.5, 0.5, 0.5, -0.5], y .+ [-0.5, -0.5, 0.5, 0.5])
            plot!(rect, fillcolor=:green, linecolor=:black, label="Study")
        elseif value == 0
            rect = Shape(x .+ [-0.5, 0.5, 0.5, -0.5], y .+ [-0.5, -0.5, 0.5, 0.5])
            plot!(rect, fillcolor=:blue, linecolor=:black, label="Work")
        end
    end
end

# Remove duplicate legend entries
# plot!(legend=:bottomright)
# Display the plot
display(p)