# Game Theory

**Game Theory** is modelling and analyzing situations where different agents make decisions, each one trying to optimize their own payoff which also depends on the choices of others. It is useful for predicting players’ joint decisions in interactive situations and designing incentives to guide the players’ joint decisions to desirable directions. The main challenge is the computational tractability.

## Nash Equilibrium

A **Nash equilibrium** is a key concept in game theory that represents a **stable state** in a strategic interaction where **no player has an incentive to deviate unilaterally from their current strategy**. A Nash equilibrium occurs when:

- **Mutual Best Responses**: Each player's strategy is optimal, given the strategies of the others.
- **No Regret**: Once the equilibrium is reached, no player regrets their choice, as switching strategies unilaterally would not yield a better outcome.
- **Existence**: In games with a finite number of players and strategies, at least one Nash equilibrium (in mixed strategies) always exists, as proven by John Nash.

https://en.wikipedia.org/wiki/Nash_equilibrium

### Examples of famous games :

- **_Prisoner's dilemma_** : The Prisoner's Dilemma illustrates how rational decision-making can lead to suboptimal outcomes when trust and communication are absent. Two prisoners are accused of a crime and interrogated separately. They cannot communicate with each other and are each given the option to cooperate (stay silent) or defect (betray).

  - The outcomes depend on their decisions: If both cooperate (stay silent): Each serves a light sentence (e.g., 1 year). If one defects and the other cooperates: The defector goes free, and the cooperator receives a heavy sentence (e.g., 10 years). If both defect: Each serves a moderate sentence (e.g., 5 years).
  - **Nash Equilibrium:** The dominant strategy for both prisoners is to defect because betraying minimizes individual risk, regardless of the other player's choice. Defection leads to a Nash equilibrium at (Defect, Defect), where both receive moderate punishment (5 years each).
  - **Pareto Inefficiency:** Mutual cooperation (1 year each) would result in a better collective outcome than mutual defection (5 years each). However, the prisoners' inability to trust each other leads to a worse outcome.
  - https://en.wikipedia.org/wiki/Prisoner%27s_dilemma

- _The beauty contest game :_ The Keynesian Beauty Contest refers to a situation where participants in a contest or decision-making process must choose not what they personally find most attractive or best, but what they believe others will find most attractive or best. In this framework:

  - Each participant tries to anticipate the average opinion or the "common consensus" rather than relying solely on their own judgment.
  - Decision-making is driven by second-order thinking (and higher) : predicting what others are likely to think, rather than acting based on personal preferences or beliefs.
  - https://en.wikipedia.org/wiki/Keynesian_beauty_contest
  - https://en.wikipedia.org/wiki/Guess_2/3_of_the_average
  - https://www.youtube.com/watch?v=j8ZVkVjDPxo&ab_channel=IntermittentDiversion

- _Public goods game :_ The Public Goods Game models situations where individuals must decide whether to contribute to a common resource (a public good) that benefits all participants, regardless of their personal contribution. A public good is characterized by:

  - Non-excludability: No one can be excluded from benefiting from it, regardless of whether they contributed (e.g., clean air, national defense).
  - Non-rivalry: One person's use of the good does not reduce its availability to others.
  - **Free-Riding Problem:** Players can benefit from the public good without contributing, incentivizing selfish behavior.
  - **Nash equilibrium:** The group's total payoff is maximized when everyone contributes all of their tokens to the public pool. However, the Nash equilibrium in this game is simply zero contributions by all; if the experiment were a purely analytical exercise in game theory it would resolve to zero contributions because any rational agent does best contributing zero, regardless of whatever anyone else does. This only holds if the multiplication factor is less than the number of players, otherwise, the Nash equilibrium is for all players to contribute all of their tokens to the public pool.
  - https://en.wikipedia.org/wiki/Public_goods_game

- _Tragedy of the commons :_ The Tragedy of the Commons is a concept in economics and environmental science that describes individuals who have a resource that is common (shared by multiple individuals) and rivalrous (its use by one person diminishes its availability for others). There are no effective regulations or mechanisms to manage the resource collectively.They are acting in their self-interest, and can overexploit and deplete a shared resource, even though this is against the long-term interest of the entire group.

  - To address the Tragedy of the Commons, mechanisms are needed to align individual incentives with collective welfare: **Regulation**: Governments can set rules to limit access or impose quotas (e.g., fishing limits, pollution controls). **Privatization**: Assigning property rights can make individuals accountable for the resources they use. **Community Management**: Local communities can develop shared norms and agreements to manage resources sustainably (as studied by Elinor Ostrom, Nobel Prize winner). **Incentives**: Economic tools like taxes, subsidies, or tradable permits can encourage sustainable behavior (e.g., carbon credits).
  - https://en.wikipedia.org/wiki/Tragedy_of_the_commons

- _Others games :_
  - Stag hunt : https://en.wikipedia.org/wiki/Stag_hunt (2 Nash equilibrium fair)
  - Chicken : https://en.wikipedia.org/wiki/Chicken_(game) (2 Nash equilibrium with opposite actions of the two players and unfair)
  - Battle of the sexes : https://en.wikipedia.org/wiki/Battle_of_the_sexes_(game_theory) (2 Nash equilibrium but unfair because one player consistently does better than the other)
  - Social dilemma games : A situation in which all individuals would be better off cooperating but fail to do so because of conflicting interests between individuals that discourage joint action

### Mixed Nash Equilibrium

On the contrary of **pure Nash equilibrium**, each player selects a Probability Distribution over Actions or **mixed strategy**. A **Mixed Nash Equilibrium** is a mixed strategy profile from which no player wants to deviate unilaterally. Every Game has at least one Equilibrium.

https://williamspaniel.com/2014/06/12/the-game-theory-of-soccer-penalty-kicks/

## Congestion game and Best-Response Dynamics

A **Congestion Game** is a type of game in game theory where players compete for shared resources, and **each player's payoff depends on the number of other players using the same resources** and not on the exact profile of others' choices. It models situations where the utility or cost of a resource increases as more players use it (eg. traffic, ressource allocation). https://en.wikipedia.org/wiki/Congestion_game

**Best Response Dynamics** refers to an iterative process where **players update their strategies by choosing their best response to the current strategies of other players**. Players take turns updating their strategy. A player’s best response is the strategy that maximizes their payoff given the strategies chosen by others. This process continues until no player wants to change their strategy, resulting in a Nash equilibrium. In congestion games (and potential games), **best response dynamics are guaranteed to converge to a Nash equilibrium** because the game has a potential function that strictly increases or decreases with each player's improvement. https://en.wikipedia.org/wiki/Best_response

There is a special class of congestion games for which the **Social Welfare (SW)** function can act as a Potential Function under specific conditions. This condition is linearity ? Or convexity ? https://en.wikipedia.org/wiki/Potential_game

## Stackelberg Games

A **Stackelberg game** is a strategic game in which players are divided into two roles: leaders and followers. The **leader commits to a strategy first**, and the **followers observe this choice and then respond optimally**. In optimization terms, a Stackelberg game is a bilevel optimization problem, where the leader solves an optimization problem to maximize or minimize their objective. The follower(s), in turn, solve their own optimization problems, taking the leader’s decision as a parameter.

- A way to solve it is to reformulate and convert the bilevel problem into a single-level problem using **the Karush-Kuhn-Tucker (KKT) conditions for the follower’s optimization problem** and the complementarity constraints to model the follower’s optimality.
- The Leader (e.g. Policy Maker) **chooses the subsidies such that the resulting strategy profile (aka equilibrium) is desirable.**
- https://en.wikipedia.org/wiki/Stackelberg_competition
- https://optimization.cbe.cornell.edu/index.php?title=Mathematical_programming_with_equilibrium_constraints
- https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions

## Mechanisme Design

**Mechanism Design** is a field within game theory and economics that focuses on **designing systems or rules** (mechanisms) that **lead to desirable outcomes**, even when individuals (agents) **act in their self-interest**. It works "in reverse" compared to traditional game theory: instead of analyzing the outcomes of given games, mechanism design constructs games to achieve specific objectives. Main Characteristics of Mechanism Design :

- **Objective-Oriented:** The goal is to achieve a socially optimal outcome, such as maximizing Social Welfare (total benefit to all participants) or minimizing System Costs (e.g., resources or inefficiencies).
- **Private Valuations:** Players (agents) have private information about their preferences, valuations, or costs over available resources. Mechanisms must account for this **asymmetry of information**.
- **Strategic Behavior:** Players act rationally to maximize their own payoff or utility. The mechanism must anticipate and incentivize this behavior to align with the designer's objectives.
- **Components of a Mechanism:**
  - **Communication Model:** Defines how agents share information (e.g., bids, preferences).
  - **Allocation Rule:** Specifies how resources are distributed among agents.
  - **Payment Rule:** Determines what each agent pays or receives, ensuring incentives align with system goals.
- A **good mechanism features:**
  - **Strong Incentive Guarantees (DSIC):** The mechanism can predict what players will do.
  - **Strong Performance Guarantees:** Minimize efficiency loss or "social welfare gap" due to strategic behavior.
  - **Computational Efficiency:** The mechanism should be implementable in polynomial time. Allocation and payment rules must be practical and scalable for real-world applications.
- https://en.wikipedia.org/wiki/Mechanism_design
- https://en.wikipedia.org/wiki/Vickrey_auction#:~:text=A%20Vickrey%20auction%20or%20sealed,is%20the%20second%2Dhighest%20bid.

### Monotone Allocation Rules and Myerson's Lemma

A **monotone allocation rule** is an allocation rule in mechanism design where an agent’s probability or share of **receiving a good (or resource) increases or remains the same as their reported valuation increases**. In simpler terms, the higher a player’s bid or valuation, the more likely they are to receive the good or resource.

**Myerson's Lemma** provides a **fundamental characterization of incentive-compatible mechanisms** in single-parameter settings. It links an agent’s allocation rule to their payment rule, ensuring that truthful bidding maximizes their utility. For a mechanism to be incentive-compatible (truthful), the following must hold:

- The **allocation rule** must be monotone.
- The **payement rule** is unique and can be derived using an explicit formula (book).

**Read 2nd and 3rd chapters of AGT book !!!**

## VCG Mechanism

We are considering a **multi-parameter mechanism design problem** where N strategic participants with a finist set W of possible outcomes and each agent i has a private valuation vi(w) for each outcome w in W. The **Vickrey-Clarke-Groves (VCG)** mechanism is a fundamental framework in mechanism design used to allocate resources or make collective decisions efficiently. It **incentivizes truthful reporting of private information** (valuations or preferences) while maximizing social welfare.

- The payment rule is the core of the VCG mechanism and defines how much each agent pays. The **payment for each agent i is based on the externality they impose on others by their presence in the system**. This payement rule will ensure that the truthfull bidding results in an outcome that maximizes agent i's utility.
- **Read 7th chapter of AGT book !!!**
- In the class exercice, the payoff is not anymore the valuation minus the cost but the payement minus the cost. In that case, the payement rule is representing a benefit for the players and they are bidding on their costs.
- https://en.wikipedia.org/wiki/Vickrey%E2%80%93Clarke%E2%80%93Groves_mechanism

## Learning in games

So far, we have been dealing with Games where equilibria can be calculated. In general, though, Nash equilibria are intractable. When players cannot reason out (or maybe they do not even know) the Game, they **learn by playing repeatedly**. We can differentiate :

- A **Nash equilibrium** occurs when every player chooses their strategy such that no player can unilaterally improve their payoff by deviating from their chosen strategy. Players make decisions independently. Each player’s strategy is a best response to the strategies of others.The solution is self-enforcing (no incentive to deviate).
- A **correlated equilibrium** allows for an external signal (or recommendation) that players can use to condition their strategies, potentially leading to better payoffs for all players compared to a Nash equilibrium. Players may coordinate their strategies based on shared information or signals. No player benefits from deviating unilaterally from the recommended strategy. It generalizes Nash equilibrium by allowing correlation between players’ strategies. _Example: Drivers at an intersection following a traffic signal recommendation (e.g., one goes while the other stops)._ https://en.wikipedia.org/wiki/Correlated_equilibrium
- A **coarse correlated equilibrium** is the most general form of equilibrium, where players agree to follow an external signal before knowing their recommendation and cannot benefit by deviating after observing it. Players commit to follow the signal distribution without analyzing specific outcomes. It encompasses correlated equilibrium and Nash equilibrium as special cases. This is often the result of learning processes or less restrictive settings. _Example: Players deciding to follow a mediator’s random allocation of resources in advance._ https://www.cs.cornell.edu/courses/cs6840/2020sp/note/lec16.pdf

We can explain how would players learn in games :

- Begin with some arbitrary strategy (first rounds)
- Evaluate the arbitrary strategy given an observed history of repeated interaction (multiple rounds of the game). A tractable and interpretable way to do it is to calculate the **Average payoff under best pure strategy** which means the results of using a **single action strategy** (every time the same decision) on the last rounds. This allows to calculate the **Regret** which represent this **average payoff under best pure strategy minus the average payoff so far** (reality of the last rounds).
- Improve the strategy using regret calculation :
  - **Regret Matching**: Choosing the action with probability proportional to its regret.
  - **Fictitious Play**: Compute the actions that the other players will do (statistic) and optimize our action in response to those previsible actions.
  - **Multiplicative Weights**: At each time step each action is chosen with probability proportional to its weight. Update weights based on the actions's reward.
- Those actions are guaranteeing **vanishing regret** and **no regret learning converges to Coarse Correlated Equilibrium.**
- **Read 17th chapter of AGT book !!!**
- https://en.wikipedia.org/wiki/Algorithmic_game_theory

## Markov Games

A **MDP** models a **single decision-maker** (an agent) interacting with an environment over time. The agent chooses actions to maximize cumulative rewards, with outcomes depending on both the current state and action. Components:

- States (S): Set of possible states the environment can be in.
- Actions (A): Set of possible actions the agent can take.
- Transition Function (T): Probability of moving to a new state s' given a state s and an action a.
- Reward Function (R): Reward received after taking action a in state s.
- Key Characteristics:
  - Single-agent framework.
  - No interaction with other agents.
  - Focused on long-term planning in stochastic environments.

**Repeated games** extend static (one-shot) games by having the same game played multiple times. Players choose strategies in each round and aim to maximize their cumulative payoffs over time. Components:

- Players (N): Set of agents interacting in the game.
- Action Sets (A): Each player i gas a set of action Ai.
- Stages : The game is played for multiple round.
- Payoff Functions (u): Payoff depends on all players’ actions in each round.
- Key Characteristics:
  - Multi-agent framework.
  - The game is static but repeated over time.
  - Strategies can involve cooperation, punishment, or exploitation based on past actions.

**Markov games** generalize both MDPs and repeated games by incorporating multiple agents interacting in a shared environment with state transitions. Components:

- Players (N): Set of agents interacting in the game.
- States (S): Shared state space across all players.
- Actions (A): Each player i has their own set of action Ai
- Transition Function (T): Determines the next state based on the current state and all players' actions.
- Reward Functions (R): Each player has their own reward function influenced by the joint actions.
- Key Characteristics:
  - Multi-agent framework like repeated games but incorporates state transitions.
  - Combines the state-dependent dynamics of MDPs with strategic interactions of repeated games.
  - Players interact both strategically (against each other) and dynamically (via the environment).

**Partially Observable** means that we can't have a direct access to the state of the system but only observations of the system. And those observations will also evolve with our decision via a transition function.

https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process
