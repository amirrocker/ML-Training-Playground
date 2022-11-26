'''
Value, state and optimality

Value was defined as total reward that is obtained from the state.
One can write: V(s) = sum(rewardAtTimestep t * discountFactor Gamma ** steps in future)
where rewardAtTimestep is the local reward at step t of the episode. (Discounted or not is
our choice)


Lets look at environment:
Consider a very simplistic environment.
The agents initial state - state 1.
The agent can go left - state 2 - reward is 1
The agent can go down - state 3 - reward is 2

The environment is always deterministic - every action always succeeds and the agent always starts at state 1.

What is the value of state 1?
Without more context this question cannot be answered. Even a simple environment such as this has a potentially
infinite amount of possible states.

- agent always goes left
- agent always goes down
- agent goes left 0.1(10%) and down 0.9(90%) of the time
- agent goes left 0.5 and down 0.5 of the time

For these four policies we can calculate the value of state 1:

- state 1 = 1.0
- state 1 = 2.0
- state 1 = 0.1 * 1.0 + 0.9 * 2.0 = 1.9
- state 1 = 0.5 * 1.0 + 0.5 * 2.0 = 1.5

Now we can deduce the optimal policy naively. Remember the goal is to accumulate the most reward. But simply
being greedy does not work. Imagine state 3 (agent goes down) to not be a terminal state but rather can be
traversed to state 4. But the reward for state 4 is -20. Therefor it is a trap for a greedy agent.

The Bellman Equation of Optimality

Imagine our agent observes state s0 and has N available actions. Every action leads to another state s1-vN.
Each state has a corresponding reward v1-vN. We also assume to know all the values, Vi, of all states connected
to s0.

Choosing an action ai and calculate the value given for this action, then the value is calculated like
Vs0(a|A)=max(sum(ri+Vi))

with discount factor:
Vs0(a|A)=max(sum(ri+(GAMMA**t)+Vi))

This looks a bit similar to previous cross-entropy impl. where we also have a 'greedy' agent.
But the bellmann equation has a significant distinction.


'''
