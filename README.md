# Sarsa_Q-learning

## SARSA
State–action–reward–state–action (SARSA) is an algorithm for learning a Markov decision process policy, used in the reinforcement learning area of machine learning. 
The main function for updating the Q-value depends on the current state of the agent S, the action the agent chooses A, the reward R the agent gets for choosing this action, the state S' that the agent enters after taking that action, and finally the next action A' the agent chooses in its new state.
 
## Q-learning
Q-learning is a model-free reinforcement learning algorithm to learn a policy 
telling an agent what action to take under what circumstances.
Q-learning finds an optimal policy in the sense of maximizing the expected 
value of the total reward over all successive steps, starting from the current 
state.
  
## Hyperparameters
To find the optimal policy, we need to tune the hyperparameters.

Episodes- The number of times the agent will run in the grid- world. In this 
exercise I try Episodes= 100,000, because we want the agent to run 
a lot of times in the grid- world in a reasonable time.

gamma- The discount factor determines the importance of future rewards. In    
this exercise I use gamma=0.9.
alpha- The learning rate determines to what extent newly acquired 
information overrides old information. In this exercise I try small alphas 
(0.01-0.09), because we want the agent to learn from all the episodes, and 
not only from his last episodes.

Epsilon decay- rate- Epsilon is used when we are selecting specific actions 
base on the Q values we already have. We randomize number between 0 to 
1, and if the Epsilon is higher than him, the agent will continue to exploration 
the grid- world (choose random action). Otherwise, the agent will choose the 
optimal action based on the Q values. 
We want the agent to explore in the beginning and go by the optimal 
actions in the end. Therefore, we use the decay- rate. Every episode we 
lower the value of the Epsilon.  

In this exercise I try three decay- rate
epsilon=1 
equation: epsilon=epsilon* decay- rate,                                                           
decay- rate =[0.9995,0.99995,0.999995] 
I chose this values of decay- rates, because we told in the lecture that- 
Epsilon= Epsilon*0.9 is the most common equation. This decay- rate stop 
the agent exploration in earlier episode than I want, and do not give us the 
optimal policy. To make the agent exploration longer, I make the decay 
function goes slower by increasing the value of the decay- rate.   

