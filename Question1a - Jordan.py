import csv
import tensorflow as tf
from numpy import arange, round, meshgrid, resize, math
import matplotlib.pyplot as plt


# def read_two_spiral_file(filename="../datasets/spiralsdataset.csv"):
#     x = []
#     y = []
#
#     with open(filename) as csv_file:
#         csv_reader = csv.reader(csv_file)
#         for row in csv_reader:
#             x.append(list(map(float, row[:-1])))
#             y.append([int(row[-1])])
#
#     return x, y
#
#
# x, y = read_two_spiral_file()


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

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print('Training...')

errors = []

# Train
for i in range(N_EPOCHS):
    _, error = sess.run([optimizer, cost], feed_dict={x_: x, y_: y})
    errors.append(error)
    if i % 1000 == 0:
        print(i, 'error count: ', error)
    if error < 0.02:
        break

plt.plot(errors)
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

x, y = meshgrid(activation_range, activation_range)
plt.scatter(x, y, c=['b' if x > 0 else 'y' for x in classifications])
plt.show()

################################################
# plt.plot(errors)
#
#
# def get_col(lst, col):
#     return [row[col] for row in lst]
#
#
# classifications = sess.run(outputs, feed_dict={x_: x})

# for input, target, prediction in zip(x, y, classifications):
#     print("input", input, "target", target, "prediction", prediction)
#
# plt.figure(2)
# plt.title("predictions, red is spiral 1 and blue is spiral 2")
# for input, prediction in zip(x,classifications):
#     if(prediction[0]<0.5):
#         plt.scatter(input[0], input[1],c = "red" )
#     else:
#         plt.scatter(input[0], input[1], c = "blue")
#
# # plt.show()
#
# plt.figure(3)
# plt.title("green points are predicted correctly")
# for input,target, prediction in zip(x,y,classifications):
#     if(round(prediction[0]) == target):
#         plt.scatter(input[0], input[1],c = "green" )
#     else:
#         plt.scatter(input[0], input[1], c = "red")
#
# plt.show()
