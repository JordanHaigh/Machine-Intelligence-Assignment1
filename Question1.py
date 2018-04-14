
# Import libraries
import tensorflow as tf
from numpy import round
import matplotlib.pyplot as plt
import csv
import math


def read_dataset(filename="spiralsdataset.txt"):
    x = []
    y = []
    spiral1 = True
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            currentRow = row

            splittedRow = currentRow[0].split(' ')
            # list cleaner

            newRow = []

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

for i in x:
    i.append(math.sin(i[0]))
    i.append(math.sin(i[1]))

labels = {0: 'spiral2', 1: 'spiral1'}

# Create data placeholders
# x_ is a placeholder for the inputs to the neural network
x_ = tf.placeholder(tf.float32, [None, 4])
# y_ is a place holder for the output of the neural network
y_ = tf.placeholder(tf.float32, [None, 1])
#
# printf(x_)
nodesInH1 = 8
nodesInH2 = 8
nodesInH3 = 1
# Create first layer weights
layer_0_weights = tf.Variable(tf.random_normal([4, nodesInH1]))
layer_0_bias = tf.Variable(tf.random_normal([nodesInH1]))
layer_0 = tf.nn.sigmoid(tf.add(tf.matmul(x_, layer_0_weights), layer_0_bias))

# Create second layer weights
layer_1_weights = tf.Variable(tf.random_normal([nodesInH1, nodesInH2]))
layer_1_bias = tf.Variable(tf.random_normal([nodesInH2]))
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_0, layer_1_weights), layer_1_bias))

# Create third layer weights
layer_2_weights = tf.Variable(tf.random_normal([nodesInH2, nodesInH3]))
layer_2_bias = tf.Variable(tf.random_normal([nodesInH3]))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, layer_2_weights), layer_2_bias))

# # Create third layer weights
# layer_3_weights = tf.Variable(tf.random_normal([nodesInH3, 1]))
# layer_3_bias = tf.Variable(tf.random_normal([1]))
# layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, layer_3_weights), layer_3_bias))

outputs = layer_2

# Define error function
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=outputs))

# Define optimizer and its task (minimise error function)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.5).minimize(cost)

N_EPOCHS = 30000

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print('Training...')

errors = []
plt.figure(1)
# Train
for i in range(N_EPOCHS):
    _, error = sess.run([optimizer, cost], feed_dict={x_: x, y_: y})
    errors.append(error)
    if (i % 1000 == 0):
        print(i)

plt.plot(errors)

def get_col(lst, col):
    return [row[col] for row in lst]


classifications = sess.run(outputs, feed_dict={x_: x})
for input, target, prediction in zip(x, y, classifications):
    print("input", input, "target", target, "prediction", prediction)

plt.figure(2)
plt.title("predictions, red is spiral 1 and blue is spiral 2")
for input, prediction in zip(x,classifications):
    if(prediction[0]<0.5):
        plt.scatter(input[0], input[1],c = "red" )
    else:
        plt.scatter(input[0], input[1], c = "blue")

# plt.show()

plt.figure(3)
plt.title("green points are predicted correctly")
for input,target, prediction in zip(x,y,classifications):
    if(round(prediction[0]) == target):
        plt.scatter(input[0], input[1],c = "green" )
    else:
        plt.scatter(input[0], input[1], c = "red")

plt.show()
