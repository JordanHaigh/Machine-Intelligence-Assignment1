import tensorflow as tf
from numpy import arange, round, meshgrid, math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def csv_to_dataframe(filename="spirals/2SpiralsRotated90.txt"):
    df = pd.read_csv(filename, header=None, names = ["x","y","spiralid"])
    x = df.iloc[:,0:2]
    y = df.iloc[:,2:3]
    return df, x, y


df, x, y = csv_to_dataframe()





labels = {1: 'spiral1', 2: 'spiral2', 3: 'spiral3', 4: 'spiral4'}
colors = {1: 'r', 2:'g', 3:'b', 4:'yellow'}
plt.figure(1)

##It works!
#for index, rows in df.iterrows():
#    xcoord = rows["x"]
#    ycoord = rows["y"]
#    spiralid = rows["spiralid"]
#    color = colors[spiralid]
#
#    plt.scatter(xcoord,ycoord, c=color)


span_of_x = 4

if span_of_x == 4:
    x["sin_x"] = np.sin(x["x"].astype(np.float64))
    x["sin_y"] = np.sin(x["y"].astype(np.float64))

# labels = {0: 'spiral2', 1: 'spiral1'}


# Create the model

x_ = tf.placeholder(tf.float32, [None, span_of_x])
y_ = tf.placeholder(tf.float32, [None, 1])

nodesInH1 = 40
nodesInH2 = 40

####################################GOOD COPY#############################################
# Create first layer weights
layer_0_weights = tf.Variable(tf.random_normal([span_of_x, nodesInH1]))
layer_0_bias = tf.Variable(tf.random_normal([nodesInH1]))
layer_0 = tf.nn.sigmoid(tf.add((tf.matmul(x_, layer_0_weights)), layer_0_bias))

# Create second layer weights
layer_1_weights = tf.Variable(tf.random_normal([nodesInH1, nodesInH2]))
layer_1_bias = tf.Variable(tf.random_normal([nodesInH2]))
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_0, layer_1_weights), layer_1_bias))

# Create third layer weights
layer_2_weights = tf.Variable(tf.random_normal([nodesInH2, 1]))
layer_2_bias = tf.Variable(tf.random_normal([1]))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, layer_2_weights), layer_2_bias))

outputs = layer_2

# Define error function
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=layer_2))







# Define optimizer and its task (minimise error function)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.5).minimize(cost)

N_EPOCHS = 50000  # Increase the number of epochs to about 50000 to get better results. This will take some time for training.

saver = tf.train.Saver()
sess = tf.InteractiveSession()

init_op = tf.global_variables_initializer()

init_op.run()


print('Training...')

errors = []

# Train
for i in range(N_EPOCHS):
    _, error = sess.run([optimizer, cost], feed_dict={x_: x[:].values.tolist(), y_: y[:].values.tolist()})
    errors.append(error)
    if i % 1000 == 0:
        print(i, 'error count: ', error)
    if error < 0.001:
        print("reached less than 0.01 > epoch: ",i, 'error count: ', error)
        break

save_path = saver.save(sess, "./tmp/question1b/model.ckpt")
print("Model saved in path: %s" % save_path)

tf.reset_default_graph()

saver.restore(sess, "./tmp/question1b/model.ckpt")
print("Model restored.")

plt.plot(errors)
plt.title("Error function for session")
plt.xlabel("Epoch")
plt.ylabel("Error percentage")
plt.show()
#
######################################################
#
# Visualise activations

activation_range = arange(-10, 10, 0.1)  # interval of [-6,6) with step size 0.1

coordinates = [(x, y) for x in activation_range for y in activation_range]

if span_of_x == 4:
    coordinates = [(x, y, math.sin(x), math.sin(y)) for x in activation_range for y in activation_range]

classifications = round(sess.run(layer_2, feed_dict={x_: coordinates}))

meshx, meshy = meshgrid(activation_range, activation_range)
plt.scatter(meshx, meshy, c=['b' if x > 0 else 'y' for x in classifications])
plt.title("Classification of Two Spiral Task")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



##################################################
classifications = round(sess.run(layer_2, feed_dict={x_: x}))

plt.title("Predictions, red is spiral 1 and blue is spiral 2")
for coord, cls in zip(x.values.tolist(),classifications):
    #print(coord[0], coord[1], cls)
    if(cls == 0):
        plt.scatter(coord[0],coord[1],c = "red" )
    else:
        plt.scatter(coord[0],coord[1],c = "b" )
plt.show()

#
#
#
plt.title("green points are predicted correctly")
for coord, target, prediction in zip(x.values.tolist(),y.values.tolist(),classifications):
    if(round(prediction[0]) == target[0]):
        plt.scatter(coord[0], coord[1],c = "green" )
    else:
        plt.scatter(coord[0], coord[1], c = "red")

plt.show()

