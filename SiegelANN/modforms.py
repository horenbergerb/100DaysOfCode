import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

filename = ""
if(len(sys.argv) == 2):
    filename = sys.argv[1]
else:
    print("Fuck off")
    exit

#get the data, remove null values, remove bad characters, cast as integers
trainFrame = pd.read_csv(filename, sep='\s+')
trainFrame = trainFrame[trainFrame.Coeff != "unknown"]
trainFrame = trainFrame.replace(',', '', regex=True)
trainFrame = trainFrame.astype(float)

#create the test frame
testFrame = trainFrame.sample(frac=.1,random_state=200)
trainFrame = trainFrame.drop(testFrame.index)

#tags for inputs and labels
inputs = ['Det', 'UnredA', 'UnredB', 'UnredC', 'X', 'Y']
labels = ['Coeff']

plt.scatter(testFrame['Det'], testFrame[labels])
plt.xlabel('Det')
plt.ylabel('Coeff')
plt.axis('equal')
plt.axis('square')

plt.show()


#statistics on the inputs of the train frame
train_stats = trainFrame[inputs].describe()
train_stats = train_stats.transpose()

print(trainFrame[labels].describe().transpose())
print(train_stats)

#norm using the inputs of the train frame
def norm(x):
    return (x-train_stats['mean'])/train_stats['std']
#norm the trainframe
normedTrainFrame = norm(trainFrame[inputs])
norm_stats = normedTrainFrame.describe().transpose()
print(norm_stats)
#norm the test frame
normedTestFrame = norm(testFrame[inputs])

#print(trainFrame[inputs])

def build_model():
    #initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)
    initializer = keras.initializers.RandomUniform(minval = -2.0, maxval = 2.0, seed=None)
    optimizer = keras.optimizers.Adam(lr=.001, beta_1=.9, beta_2=.999, epsilon=1e-8)
    model = keras.Sequential([
        layers.Dense(32, kernel_initializer=initializer, use_bias=True, activation=tf.nn.relu, input_shape=[len(normedTrainFrame.keys())]),
        layers.Dense(32, kernel_initializer=initializer, use_bias=True, activation=tf.nn.relu),
        layers.Dense(32, kernel_initializer=initializer, use_bias=True, activation=tf.nn.relu),
        layers.Dense(32, kernel_initializer=initializer, use_bias=True, activation=tf.nn.relu),
        layers.Dense(32, kernel_initializer=initializer, use_bias=True, activation=tf.nn.relu),
        layers.Dense(32, kernel_initializer=initializer, use_bias=True, activation=tf.nn.relu),
        layers.Dense(32, kernel_initializer=initializer, use_bias=True, activation=tf.nn.relu),
        layers.Dense(1, kernel_initializer=initializer, activation='linear')
        ])

    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error'])
    return model

model = build_model()
#print(model.get_weights())

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch%10 == 0: print('')
        print('.', end = '')

history = model.fit(
    normedTrainFrame, trainFrame[labels],
    epochs=200, validation_split = .1, verbose = 0,
    callbacks=[PrintDot()])

#print(history.history)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Validation Error')
    plt.legend()
    plt.show()

plot_history(history)

#loss, mae, mse = model.evaluate(testFrame[inputs], testFrame[labels], verbose=0)

test_predictions = model.predict(normedTestFrame).flatten()

plt.scatter(testFrame[labels], test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')

_ = plt.plot([-100, 100], [-100, 100])

plt.show()

#sess=tf.Session()
#init=tf.global_variables_initializer()
#sess.run(init)

#print(sess.run(y_pred))

#loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
#optimizer = tf.train.GradientDescentOptimizer(0.01)
#train = optimizer.minimize(loss)

#for i in range(1000):
    #first part seems to absorb some extra data about types
#    _, loss_value = sess.run((train, loss))

#print(sess.run(y_pred))
