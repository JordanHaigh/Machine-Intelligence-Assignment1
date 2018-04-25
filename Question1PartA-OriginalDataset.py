import csv
import tensorflow as tf
from numpy import arange, round, meshgrid, resize, math
import matplotlib.pyplot as plt

def read_dataset(filename="spirals/spiralsdataset.txt"):
    x = []
    y = []
    spiral1 = True
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            currentRow = row

            splittedRow = currentRow[0].split(' ')
            # list cleaner

            newRow = [];

            for i in range(len(splittedRow)):
                if (not splittedRow[i] == ""):
                    list.append(newRow, splittedRow[i])

            # print("New row ", newRow)

            x.append(list(map(float, newRow[:2])))

            if (spiral1):
                temp = []
                temp.append(1)
                y.append(temp)
            else:
                temp = []
                temp.append(0)
                y.append(temp)
                # y.append(list(0))
            spiral1 = not spiral1
            # y.append(int(row[-1]))
    # print("any decriptor ", x[0])

    return x, y


x, y = read_dataset()

span_of_x = 4

if span_of_x == 4:
    for i in x:
        i.append(math.sin(i[0]))
        i.append(math.sin(i[1]))

# labels = {0: 'spiral2', 1: 'spiral1'}


# Create the model

x_ = tf.placeholder(tf.float32, [None, span_of_x])
y_ = tf.placeholder(tf.float32, [None, 1])

nodesInH1 = 40
nodesInH2 = 40

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
    _, error = sess.run([optimizer, cost], feed_dict={x_: x, y_: y})
    errors.append(error)
    if i % 1000 == 0:
        print(i, 'error count: ', error)
    if error < 0.01:
        print("reached less than 0.01 > epoch: ",i, 'error count: ', error)
        break
save_path = saver.save(sess, "./tmp/question1a/model.ckpt")
print("Model saved in path: %s" % save_path)

tf.reset_default_graph()

saver.restore(sess, "./tmp/question1a/model.ckpt")
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
activation_range = arange(-6, 6, 0.1)  # interval of [-6,6) with step size 0.1
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

################################################
#

classifications = round(sess.run(layer_2, feed_dict={x_: x}))

plt.title("Predictions, red is spiral 1 and blue is spiral 2")
for coord, cls in zip(x,classifications):
    #print(coord[0], coord[1], cls)
    if(cls == 0):
        plt.scatter(coord[0],coord[1],c = "red" )
    else:
        plt.scatter(coord[0],coord[1],c = "blue" )
plt.show()




plt.title("green points are predicted correctly")
for coord, target, prediction in zip(x,y,classifications):
    if(round(prediction[0]) == target[0]):
        plt.scatter(coord[0], coord[1],c = "green" )
    else:
        plt.scatter(coord[0], coord[1], c = "red")

plt.show()


####################################################
#

#https://www.tensorflow.org/programmers_guide/saved_model
## Create some variables.
#v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
#v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)
#
#inc_v1 = v1.assign(v1+1)
#dec_v2 = v2.assign(v2-1)
#
## Add an op to initialize the variables.
#
#
## Later, launch the model, initialize the variables, do some work, and save the
## variables to disk.
#with tf.Session() as sess:
#  sess.run(init_op)
#  # Do some work with the model.
  # Save the variables to disk.
  
  ###################################################

## Create some variables.
#v1 = tf.get_variable("v1", shape=[3])
#v2 = tf.get_variable("v2", shape=[5])
#
## Add ops to save and restore all the variables.
#saver = tf.train.Saver()
#
## Later, launch the model, use the saver to restore variables from disk, and
## do some work with the model.
#with tf.Session() as sess:
#  # Restore variables from disk.
#  saver.restore(sess, "/tmp/model.ckpt")
#  print("Model restored.")
#  # Check the values of the variables
#  print("v1 : %s" % v1.eval())
#  print("v2 : %s" % v2.eval())
  ################################################
#  
#  saver.restore(sess, "/tmp/model.ckpt")
#  print("Model restored.")