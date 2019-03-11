import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make("MsPacman-ram-v0")

##Functions to build:
#Neural Net constructor
#Training initalizer
#training data updater
#training function
#Playing function
#Generation loop

#########
##Training Initializer
#Plays randomly for a few games
#Gives the "seed" for future data

##Training data updater
#Plays games using the net
#Stores observation frames, corresponding action and resulting reward

##Training function
#Updates weights based on the training data

##Playing function
#Makes decisions using the net
#Returns the results of the decision

##Generation loop
#Uses the training function to generate a dataset
#Plays games and replaces low-scoring games in the dataset with
#higher datasets

##########
##Neural net constructor
#Takes previous data-frame and its respective actions/rewards
#Also takes an arbitrary action
#Predicts the reward of taking that action
#Practical use will regress it on all eight possible actions
#Then pick the one with the highest reward

initial_games = 200
avg_scores = []

def training_data_initializer():
    #will store the actions taken based on observations for the neural net
    env.reset()
    raw_data = []
    scores = []
    for game_index in range(initial_games):
        game_memory = []
        done = 0
        score = 0
        while not done:
            #do a random action and record the outputs
            action = random.randrange(0, env.action_space.n)
            observation, reward, done, info = env.step(action)
            #print(observation)
            score += reward
            raw_data.append([observation, action, reward])
        env.reset()
        scores.append(score)
        #tracking the amount of games gone by
        if game_index % 10 == 0:
            print("{} games simulated...".format(game_index))

    avg_scores.append(sum(scores)/len(scores))
    return raw_data

def build_model(input_size):
    model = Sequential()

    model.add(Dense(128, input_dim = input_size, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

def training_function(training_data, model):
    X = []
    Y = []
    for move in training_data:
        #print(move)
        #print("BOOM")
        X.append(np.append(move[0], move[1]).flatten())
        Y.append(move[2])
    #print(X)
    #print("bang")
    model.fit(np.array(X), np.array(Y), epochs = 20)
    return model

def make_decision(observation, model):
    #print(observation)
    #print("WOW")
    decisions = [0.0]*8
    for i in range(0, 8):
        decisions[i] = model.predict(np.array([np.append(observation, i).flatten()]))
    #print(decisions)
    return np.argmax(decisions)

def play_game(model):
    new_data = []
    done = 0
    score = 0
    prev_obs = []
    env.reset()

    while not done:
        env.render()
        if len(prev_obs) < 1:
            action = random.randrange(0, env.action_space.n)
        else:
            action = make_decision(prev_obs, model)

        observation, reward, done, info = env.step(action)
        prev_obs = observation
        score += reward
        new_data.append([observation, action, reward])
    env.reset()
    return (score, new_data)
        

training_data = training_data_initializer()

#print(np.append(training_data[0][0], training_data[0][1]).flatten())
input_size = np.append(training_data[0][0], training_data[0][1]).flatten().size
print("Input size: {}".format(input_size)) 

#Figure out where to put this
model = build_model(input_size)
model = training_function(training_data, model)
scores = []
for i in range(100):
   scores.append(play_game(model)[0])
avg_scores.append(sum(scores)/len(scores))

print(avg_scores)
