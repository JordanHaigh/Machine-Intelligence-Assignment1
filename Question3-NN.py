import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_digits

digits = load_digits()

#flattens a 2d array into a list
def flatten(image):
    temp = []
    for sublist in image:
        for element in sublist:
            temp.append(element)

    return temp

#converts and integer into an array with a 1 in the parameters position
def redHot(num):
    temp = [0,0,0,0,0,0,0,0,0,0]
    temp[num] = 1
    print(num, temp)
    return temp


##shows a digit picture
# import pylab as pl
# pl.gray()
# pl.matshow(digits.images[0])
# pl.show()


# get the inputs and labels for the neural network
imgs = digits.images
labels = digits.target
x = []
y = []

for i in range(len(imgs)):
    x.append(flatten(imgs[i]))
    y.append(redHot(labels[i]))



#setup place holders
x_ = tf.placeholder(tf.float32, [None, 64 ])
y_ = tf.placeholder(tf.float32, [None, 10])

nodesInH1 = 100
nodesInH2 = 100
nodesInH3 = 10

# Create first layer weights
layer_0_weights = tf.Variable(tf.random_normal([64, nodesInH1]))
layer_0_bias = tf.Variable(tf.random_normal([nodesInH1]))
layer_0 = tf.nn.sigmoid(tf.add((tf.matmul(x_, layer_0_weights)), layer_0_bias))

# Create second layer weights
layer_1_weights = tf.Variable(tf.random_normal([nodesInH1, nodesInH2]))
layer_1_bias = tf.Variable(tf.random_normal([nodesInH2]))
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_0, layer_1_weights), layer_1_bias))

# Create third layer weights
layer_2_weights = tf.Variable(tf.random_normal([nodesInH2, nodesInH3]))
layer_2_bias = tf.Variable(tf.random_normal([nodesInH3]))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, layer_2_weights), layer_2_bias))

# Create fourth layer weights and bias
# layer_3_weights = tf.Variable(tf.random_normal([nodesInH3, 10]))
# layer_3_bias = tf.Variable(tf.random_normal([10]))
# layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, layer_3_weights), layer_3_bias))
outputs = layer_2

# Define error function
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=outputs))

# Define optimizer and its task (minimise error function)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.5).minimize(cost)

N_EPOCHS =50000  # Increase the number of epochs to about 50000 to get better results. This will take some time for training.

# open session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print('Training...')

errors = []

# Train
for i in range(N_EPOCHS):
    _, error = sess.run([optimizer, cost], feed_dict={x_: x, y_: y})
    errors.append(error)
    if i % 100 == 0:
        print(i, 'error count: ', error)
    if error < 0.01:
        print("reached less than 0.01 > epoch: ",i, 'error count: ', error)
        break


# get outputs from the now trained NN
classifications = sess.run(outputs, feed_dict={x_: x})
# get the correct answers
correctOutputs = tf.equal(tf.argmax(classifications,1), tf.argmax(y,1))
# cast the boolean array into a float array then get the average result
# correct answers will be cast to a 1 and incorrect a 0 so the average will be the percentage correct
accuracy = tf.reduce_mean(tf.cast(correctOutputs, 'float'))

print("Accuracy:", accuracy.eval())


# plot error function
plt.plot(errors)
plt.title("Error function for session")
plt.xlabel("Epoch")
plt.ylabel("Error percentage")
plt.show()


