import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#A rudimentary ML model which trains on a set of randomly generated games. Highly limited in that it emulates actions above a certain reward threshold. Explores introductory ML concepts and integrates
#popular ML tech

env = gym.make("MsPacman-ram-v0")

#Just getting info about our space
print("Total actions:")
print(env.action_space.n)

print("Action meanings:")
print(env.env.get_action_meanings())

print("Observation space:")
print(env.observation_space)

env = gym.make('MsPacman-v0')
env.reset()

goal_steps = 500
score_requirement = 100
initial_games = 20


def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, env.action_space.n)
            observation, reward, done, info = env.step(action)

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                output = [0]*env.action_space.n
                output[data[1]] = 1
                training_data.append([data[0], output])
        env.reset()
        if game_index % 10 == 0:
            print("Bam")
    print(accepted_scores)
    return training_data


def build_model(input_size, output_size):
    model = Sequential()
    #model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(len(training_data[0][0])*len(training_data[0][0][0])*len(training_data[0][0][0][0]))))
    model.add(Dense(128, input_dim = input_size, activation='relu'))
    model.add(Dense(128, input_dim = input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

def train_model(training_data):
    #print(training_data)
    #print(len(training_data))
    #print(len(training_data[0]))
    print(training_data[0][0])
    print(training_data[0][1])

    #our input is a flattened array of all the pixels on the screen!
    #our output is an action
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0])*len(training_data[0][0][0])*len(training_data[0][0][0][0]))
    print(len(X[0]))
    Y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))

    print(len(X))
    print(len(Y))

    model = build_model(input_size=len(X[0]), output_size=len(Y[0]))

    model.fit(X, Y, epochs=10)
    return model

training_data = model_data_preparation()
trained_model = train_model(training_data)

scores = []
choices = []

for each_game in range(100):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, env.action_space.n)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(training_data[0][0])*len(training_data[0][0][0])*len(training_data[0][0][0][0]))))

        choices.append(action)
        new_observation, reward, done, infor = env.step(action)
        prev_obs = new_observation
        score += reward
        if done:
            break
    env.reset()
    scores.append(score)

print(scores)
print('Average score: ', str(sum(scores)/len(scores)))

