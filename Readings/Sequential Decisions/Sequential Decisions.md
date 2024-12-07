# Sequential Decisions

**Sequential decision-making** in computer science refers to a process where decisions are made over time in a sequence, often in environments with uncertainty or changing conditions. Each decision influences future states, outcomes, and subsequent choices, making it essential to plan strategically and adapt dynamically. Key Elements:

- **States**: The current situation or configuration of the system. States often encapsulate all the information needed to make decisions.
- **Actions**: Choices or interventions available at each step. Actions transition the system from one state to another.
- **Transition Dynamics**: Rules or probabilities governing how actions affect state changes. These can be deterministic or stochastic. A **stochastic transition function** is a mathematical model that describes the probabilities of transitioning between states in a system, given a specific action.
- **Rewards/Costs**: Metrics that evaluate the immediate value or expense of taking a particular action in a specific state.
- **Horizon**: The time span over which decisions are made. This can be finite (a fixed number of steps) or infinite (ongoing decision-making).
- **Policy**: A strategy or rule defining what actions to take in each possible state to maximize long-term rewards or minimize costs.
- **Objectives**: Goals that guide the decision-making process, typically involving optimization of cumulative rewards or minimizing long-term risks.
- A **state variable** in the context of sequential decision-making is a variable that captures **all the relevant information about the current state of the system** necessary to make decisions and predict future states. Together, the state variables **define the state of the system** at any given time. _(eg. Education level at time t but it could also be that we add the money level as a state variable)_

https://en.wikipedia.org/wiki/Sequential_decision_making

## Markov Decision Process

**Markov decision process (MDP)**, also called a **stochastic dynamic program** or stochastic control problem, is a model for sequential decision making when outcomes are uncertain. The process satisfies the **Markov property**, meaning that the next state s' depends only on the current state s and action a, not on the history of previous states or actions.

- In the next part, we are going to describe way to solve the MDP.
- **Look at course : Decision Making under Uncertainty for a way to solve a MDP -> Stochastic Optimization**
- https://en.wikipedia.org/wiki/Markov_decision_process

## Deterministic Optimization (Non linear and non stochastic)

Using **deterministic optimization** to solve a Markov Decision Process (MDP) involves **treating the stochastic transition function as if it were deterministic**, often by assuming the most likely outcome or using the expected value of transitions.

- Using this method has the **advantages** of simplifying the problem, faster computation (LP, IP), scalability, is useful for initial exploration and can be sufficient in low uncertainty environments.
- But is has the **disadvantages** of ignoring the uncertainty leading to policies that may perform poorly under uncertainty. It is taking myopic decisions and might missed opportunities because of its lack of robustness and poor handling of rare events.
- https://en.wikipedia.org/wiki/Deterministic_global_optimization

## Value Function

The **Value function V(x)** represents the maximum reward that can be achieved **from a given state x onwards**. It quantifies the future potential rewards from each state : V(x) = max_u {R(x,u) + gamma\*V(x')}. If we have the value function for each and every state, we can **derive the optimal policy** by looking at the action u as the argmax of the function.

- Since we are using stochastic transition functions, we need to **include the probabilities to arrive to a certain state x' when taking an action u at a state x**. Therefore, the value functions of the different possible states x' accessible from a state x are probability summed and multiply by gamma in the computation of the value function for state x.
- We can also define an **action value function Q(x,u)** at state x and taking the action u that describes R(x,u) + gamma\*V(x') the possible value of the value function if u was the argmax. This can be used to calculate the optimal policy by taking the action related to the highest action value function for each state.
- https://en.wikipedia.org/wiki/Value_function

### Linear Programming

The value function for each state x can be computed by **linear programming**. We are minimizing the sum of the value function for each state and making sure to find the optimal value function V\*(x) by adding the Bellman equation as a constraint (value function greater or equal than the direct reward + the probabilistic sum of the value functions of the accessible states x').

- This guarantees a globally optimal solution and can handle efficiently problems **if the state-action space if finite and small**. Moreover, it can **naturally incorporates constraints** _(eg. money constraint)_ if needed.
- But it computationally expensive for large state-action spaces and **requires explicit knowledge of the transition probabilities and rewards**.

### Dynamic Programming and backward induction

Another method is to use **dynamic programming** to compute the value function in each state. It is solving iteratively using recursive behaviour of the belmman equation. This method can be used in finite-horizon MDP.

- We start by **computing the value of the final stage** _(eg. all education level at last stage t)_. This is usually easy to compute because we know what is the best action to take to maximize our reward at the last stage. _(eg. always work to earn money)_.
- From this stage, we **use the bellman equation** to compute the value fonction of previous stages _(eg. lower education level at lower stages)_. By knowing the format of the transition function, it is possible to know which state could have been used to arrive to which state and therefore it is possible to use the stochastic transition functions.
- This method has the advantages of giving the **exact solution** and do not need a full policy representation are we are going to compute the policy iteratively for each time step.
- But it has the disadvantages of working **only for finite-horizon problem** and it requores explicit knowledge of transition probabilities and reward which might not be the case in real life. It can also becom computationally expensive for large state spaces.
- https://en.wikipedia.org/wiki/Dynamic_programming

## Value iteration and Policy iteration

In **reinforcment learning** it is common to use a value iteration method or a policy iteration method. They are using **dynamic programming** algorithm to compute the policy and/or the value function.

- In **Value iteration** we compute the optimal state value function by iteratively updating the estimate V(x). We start by initializing the value V(xt) to random values and then for each xt we update V(xt) using the Bellman equation. We do so until the value function is reaching convergence.
  - This method is easy to understand, even if stopped early it will give a good approximation and can yield a usable policy. It does not need to evaluate the policy **but the policy extraction is separate which makes it less readable**. It is memory intensive.
- In **Policy iteration** we start by choosing an arbitraty policy. Then we iteratively evaluate and improve the policy until convergence. So we initialize the actions u(xt) to random values. We start by evaluating the policy by computing for each xt the value function under the policy. Then for each xt we improve the policy by settung the ut(xt) argmax of the bellman equation.
  - It is a direct policy optimization, very efficient for small actions spaces with rapid convergence. But the policy evaluation is quite expensive and the method might be sensitive to initial policy.
- Compare to backward induction, those method **works for an infinite horizon** and scales better for large action spaces.
- Both algorithm are **converging to optimality** but policy iteration is known to converge faster than value iteration.
- https://www.baeldung.com/cs/ml-value-iteration-vs-policy-iteration

## Policy Function Approximation

The main problem of all the techniques we have seen from now is that **they need to store/represent the value/Q/optimal action for each state**. The number of states grows linearly with the discretization granularity but **grows exponentially in the number of state variables**. The **curse of dimensionality** in the context of sequential decision-making refers to the exponential growth in computational complexity as the dimensionality of the problem increases. _(eg. number of states of a chess game 10^120)_ https://en.wikipedia.org/wiki/Curse_of_dimensionality

One solution to this issue is to **approximate the policy**. The main idea is to impose a **parameterized functional form of the policy** and then to tune the parameters. _(eg. study function of education level and stage / study depending on the stage / study as a neural network)._ To tune the parameters we can use intuition, domain expertise, simulation or learning. In the case of learning, we need to use istances for which we know the optimal decision _(we know the policy for a special case, or the discretized case for eg.)._ Then we **solve the optimization problem that is minimizing the error between the know policy and the parametrized policy** to find the good parameters.

- Simplicity of applying the policy but **difficulty of tuning the policy**.
- This method allows for diverse policy representation that can capture highly non-linear mappings and is bypassing the need to compute or store a value function, reducing the complexity.
- But this method can lead to **sup-optimal policy** and we need data to use it, which is not always available.

## Value Function Approximation

Another way of solving our dimensionality problem is to use the **Value Function Approximation** method. Indeed, it is not always possible to identify some states x\* for which we know the optimal decisions and we can't always exploit monotonicity _(the policy's behavior consistently increases or decreases as certain parameters or states evolve)._ The goal is therefore to **impose a parameterized functional form of the value function V** and to tune the parameters to represents as well as possible the problem.

The first step is to identify a functional form for the value function that is depending on the states. Then we are going to **simulate random-actions trajectories using the transition function** _(we take a random state, a random action and we calculate the return and the next state)._ Then we update the parameters using a **gradient decent** with L(theta) the bellman error. The **Bellman error** represent the square of the difference between the value function at state xt and the return plus the value function at state x't+1. At the end, we are using the approximate policy to design the approximately optimal policy.

- The **advantages** of this method are the scalability and the flexibility in the representation. Approximating the value function can guide exploration by identifying promising regions of the state space, especially in problems where exhaustive exploration is infeasible.
- But it has the **disadvantages** of being computationaly complexe to train and might lead to a large number of approximation errors. It might be unstable and have convergence issues. Moreover, the lack of interpretability and the reliance on the chosen parametric forms makes it unusable sometimes.
- https://www.davidsilver.uk/wp-content/uploads/2020/03/FA.pdf

## Direct Lookahead and Cost Function Approximation

By extending the definition of the value function, we can start to understand that the problem can be seen as an optimization problem including multiple optimization problem, the decision at a stage t+1 depends on the value of the state xt+1 and we can expend the probabilistic optimization to a **single optimization problem seen as the direct lookahead**. We are coming back to stochastic programming where we conceptualize the education rate as a random variable. We can generate scenarios and introduce non-anticipativity constraints as seen in the class **Decision Making Under Uncertainty.**

- **See class Decision Making Under Uncertainty**
