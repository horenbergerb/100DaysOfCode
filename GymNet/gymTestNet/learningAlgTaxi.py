import gym
import numpy as np

env = gym.make("Taxi-v2")

#Algorithm memory
#This is a table which stores all combos of states and actions
#The entered values will be associated rewards with these combos
Q = np.zeros([env.observation_space.n, env.action_space.n])

#Total accumulated reward for each session
G = 0

#Learning rate
alpha = 0.618

for episode in range(1, 1001):
    #initialize the session
    done = False
    G, reward = 0, 0
    state = env.reset()
    #main loop
    counter = 0
    while done != True:
        counter += 1
        #Do the most valuable action available
        action = np.argmax(Q[state])
        #Collect the info of that action
        state2, reward, done, info = env.step(action)
        #Update the reward for this particular action
        #Add the actual reward plus the new potential reward (in our new state)
        #minus the previous total reward?
        Q[state, action] += alpha*(reward+np.max(Q[state2])-Q[state, action])
        G += reward
        state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))
        print('Moves taken to finish: {}'.format(counter))
