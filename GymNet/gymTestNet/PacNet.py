import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#TO DO:
#Want to input time series data
#Essentially convolution!
#Must make sure disjoint games don't overlap in training

##TIME SERIES###
#Almost there
#I need to follow the data flow and make sure training_data isn't compromised
#Separating games for training purposes would be rad
#Specifically frames between games might confuse AI
#You can do it!


#Amount of frames remembered by the AI
MEM_SIZE = 3
num_epochs = 10

env = gym.make("MsPacman-ram-v0")

#Games generated during random initialization
initial_games = 100
#number of epochs. Increases by 5 with each iteration
avg_scores = []

def training_data_initializer():
    global initial_games
    global avg_scores
    #will store the actions taken based on observations for the neural net
    env.reset()
    raw_data = []
    scores = []
    for game_index in range(initial_games):
        game_memory = []
        done = 0
        score = 0
        prev_obs = []
        counter = -1
        while not done:
            #do a random action and record the outputs
            action = random.randrange(0, env.action_space.n)
            observation, reward, done, info = env.step(action)
            score += reward
            #if we have had an observation before, append that one to our data set and the action that we took from that state
            if len(prev_obs) > 0:
                raw_data.append([prev_obs, action])
            #set prev_obs to our new state
            prev_obs = observation
            counter += 1
        env.reset()
        #append the final game score to all of the moves from that game (our metric)
        for x in range(0, counter):
            raw_data[len(raw_data)-1-x].append(score)
        scores.append(score)
        #tracking the amount of games gone by
        if game_index % 10 == 0:
            print("{} games simulated...".format(game_index))
    #tracking average scorese
    avg_scores.append(sum(scores)/len(scores))
    print(avg_scores)

    return raw_data

#construction of the model
def build_model(input_size):
    #note sure what this means (oops)
    model = Sequential()

    #some dense layers. last layer is linear. not sure how we might optimize this
    model.add(Dense(input_size, input_dim = input_size, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

#the function which takes training data and feeds it to the model for backprop
def training_function(training_data, model):

    global num_epochs
    
    X = []
    Y = []
    #taking all the moves we've stored
    for moveID in range(2, len(training_data)):
        nextSeries = []
        #flatten three frames and their actions into the next sample
        for prevMove in range(0, MEM_SIZE):
            nextSeries.extend(training_data[moveID-prevMove][0].flatten())
            nextSeries.append(training_data[moveID-prevMove][1])
        #print(np.array(nextSeries))
        X.append(nextSeries)
        Y.append(training_data[moveID][2])
    model.fit(np.array(X), np.array(Y), epochs = 20, batch_size = 1000)

    num_epochs += 3
    
    return model

#uses the model to guess which move will give the best total game score
def make_decision(observation, model):
    #it predicts the total game store for each possible action to take
    decisions = [0.0]*8
    for i in range(0, 8):
        #for each action, plug in our set of observations with an appended action
        nextDec = np.asarray([np.append(observation, i).flatten()])
        decisions[i] = model.predict(nextDec)
        #not sure why I was trying to pop this...
        #np.pop(observation)
        
    #adding some randomness
    if(random.randrange(0,10) >= 2):
        return np.argmax(decisions)
    else:
        decisions[np.argmax(decisions)] = 0.0
        return np.argmax(decisions)

#this plays a game using the model and the make_decision call
def play_game(model, renderBool):
    global MEM_SIZE
    
    new_data = []
    done = 0
    score = 0
    prev_obs = []
    memory = []
    #why is this -1?
    counter = -1
    env.reset()

    while not done:
        #render every 10th game
        if renderBool%30 == 1:
            env.render()
        #act randomly if we don't have any info
        if len(new_data) < MEM_SIZE-1:
            action = random.randrange(0, env.action_space.n)
        #otherwise make a decision using the previous observations
        else:
            action = make_decision(memory, model)
            memory = memory[129:]

        observation, reward, done, info = env.step(action)

        score += reward

        #if this isn't our first frame, record the observation we used and the decision we made
        #also add the current state and action just taken to the memory
        if len(prev_obs) > 0:
            new_data.append([prev_obs, action])
            memory = np.append(memory, action)
        #if it is our first time, add to local memory our current state

        memory = np.concatenate((memory, observation))


        prev_obs = observation

        counter += 1

    for x in range(0, counter):
        new_data[len(new_data)-1-x].append(score)
    env.reset()
    return (score, new_data)
        
def train_generation(model, training_data):
    global avg_scores
    
    gen_data = []
    scores = []
    for i in range(0, 100):
        gameScore, gameData = play_game(model, i)
        scores.append(gameScore)
        gen_data.extend(gameData)
        print("{} games simulated...".format(i+1))

    avg_scores.append(sum(scores)/len(scores))
    print(avg_scores)

    training_data.extend(gen_data)
    #print(training_data)
    model = training_function(training_data, model)
    print(avg_scores)

    return model, training_data

training_data = training_data_initializer()

input_size = (len(training_data[0][0].flatten())+1)*MEM_SIZE
print("Input size: {}".format(input_size)) 
#print(training_data[0][0])

#Figure out where to put this
model = build_model(input_size)
model = training_function(training_data, model)
for x in range(0, 100):
    model, training_data = train_generation(model, training_data)
    #print(avg_scores)
    f = open("averageScoreLog", "w")
    f.write(str(avg_scores[len(avg_scores)-1]) + " ")
