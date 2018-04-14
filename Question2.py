
# Import libraries
import tensorflow as tf
from numpy import round
import matplotlib.pyplot as plt
import csv
import math


def read_dataset(filename="OurSpirals.csv"):
    x = []
    y = []
    spiralNo = -1
    spiral1 = True
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            currentRow = row

            splittedRow = currentRow[0].split('\t')
            # list cleaner
    x = []
    y = []
    spiralNo = -1
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            currentRow = row

            splitRow = currentRow[0].split('\t')
            # list cleaner

            newRow = []

            for i in range(len(splitRow)):
                if not splitRow[i] == "":
                    list.append(newRow, splitRow[i])

            # print("New row ", newRow)

            x.append(list(map(float, newRow[:2])))

            y.append(spiralNo)

            if len(x) % 259 == 0:
                spiralNo += 1
    return x, y


x, y = read_dataset()
print(y)

# for i in x:
#     i.append(math.sin(i[0]))
#     i.append(math.sin(i[1]))

labels = {-1 : 'spiral1', 0: 'spiral2', 1: 'spiral3'}
plt.figure(1)

for i, j in zip(x, y):
    if j == -1:
        plt.scatter(i[0], i[1], c='red')
    if j == 0:
        plt.scatter(i[0], i[1], c="blue")
    if j == 1:
        plt.scatter(i[0], i[1], c="green")

#plt.show()

# Create data placeholders
# x_ is a placeholder for the inputs to the neural network
x_ = tf.placeholder(tf.float32, [None, 2])
# y_ is a place holder for the output of the neural network
y_ = tf.placeholder(tf.float32, [None, 1])
#
# printf(x_)
nodesInH1 = 8
nodesInH2 = 8
nodesInH3 = 1
# Create first layer weights
layer_0_weights = tf.Variable(tf.random_normal([2, nodesInH1]))
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

N_EPOCHS = 5000

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

# plt.show()

# assign colours so we can see 2 distinct spirals, blue for spiral 2, red for spiral 1 because we ae barbarians who do things back to front
# colours = {0: 'blue', 1: 'red'}
#
# colour_map = [colours[i] for i in x]  # map all outputs to a colour for plotting


def get_col(lst, col):
    return [row[col] for row in lst]

# def plotData(x, y, colour_map, title, xLabel, yLabel)
#     current = 0
#     while len(x) > current:
#         newX = x[current]
#         newY = y[current]
#

# def plot_iris_data(x, y, colour_map, cls, title, xlabel, ylabel):
#     plt.clf()
#     plt.figure(2, figsize=(10, 6))
#
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#
#     # Plot the training points
#     for x_, y_, colour, c in zip(x, y, colour_map, cls):
#         plt.scatter(x_, y_, c=colour, label=labels[c])
#
#     handles, label = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(label, handles))
#     plt.legend(by_label.values(), by_label.keys())
#
# #     plt.show()
# #
# #
# # plot_iris_data(get_col(x, 0), get_col(x, 1), colour_map, y, "Sepal data", "Sepal length", "Sepal width")
# #
# # plot_iris_data(get_col(x, 2), get_col(x, 3), colour_map, y, "Petal data", "Petal length", "Petal width")
# Display predictions


classifications = sess.run(outputs, feed_dict={x_: x})
for input, target, prediction in zip(x, y, classifications):
    print("input", input, "target", target, "prediction", prediction)

plt.figure(2)
plt.title("predictions, red is spiral 1 and blue is spiral 2")
for input, prediction in zip(x,classifications):
    if(prediction[0]<-0.6):
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
