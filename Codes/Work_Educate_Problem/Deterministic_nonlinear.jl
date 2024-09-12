############## Deterministic optimization ##############

using JuMP, Gurobi, Plots

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

model = Model(Gurobi.Optimizer)

@variable(model, w[1:n], Bin)
@variable(model, s[1:n], Bin)
@variable(model, e[1:n])
@variable(model, m[1:n])
@variable(model, p[1:n])

@objective(model, Max, sum(p[i] - expenses for i = 1:n))

# Payments constraint
@constraint(model, [i = 2:n], p[i] == w[i-1] * (base_salary * (1 + (e[i-1] / 2))))
@constraint(model, p[1] == 0)
# Education level
@constraint(model, [i = 2:n], e[i] == e[i-1] + s[i-1] * education_rate)
@constraint(model, e[1] == 0)
# Money constraint
@constraint(model, [i = 2:n], m[i] == m[i-1] + p[i] - expenses)
@constraint(model, m[1] == base_money)
@constraint(model, [i = 1:n], m[i] >= 0)
# Only one choice 
@constraint(model, [i = 1:n], w[i] + s[i] <= 1)

JuMP.optimize!(model)

work = value.(w)
study = value.(s)
money = value.(m)
education = value.(e)

println("Objective is: ", JuMP.objective_value(model))
println("Solution is: ")
println(work)
println(study)
println(money)
println(education)

# Time (x-axis)
time = 1:n

# Create the plot
p = plot(time, work, label="Work", marker=:circle, xlabel="Time", ylabel="Value", legend=:top)
plot!(p, time, study, label="Study", marker=:star5)
plot!(p, time, money, label="Money", marker=:diamond)
plot!(p, time, education, label="Education", marker=:cross)

# Display the plot
display(p)

### Comments ###
# Non linearity constraint in the constraint of the transition function