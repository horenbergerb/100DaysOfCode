import gym

env = gym.make("Taxi-v2")

env.reset()

print("Possible states:")
print(env.observation_space.n)
print("Current state:")
print(env.action_space.n)

print("Environment")
env.render()

env.env.s=114
print("Environment after state change:")
env.render()

print("Results of stepping:")
print(env.step(1))
print("state, reward, done, info=env.step(1)")

print("Environment after step:")
env.render()

#Now we reset and create a random agent to actually try and solve this
state = env.reset()
counter = 0
reward = None
#The reward for successfully dropping off a person is 20pts
while reward != 20:
    state, reward, done, info = env.step(env.action_space.sample())
    counter += 1
print("Steps taken to solve: " + str(counter))

