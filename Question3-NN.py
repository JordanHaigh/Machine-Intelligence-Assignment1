import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)


x_ = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float')


nodesInH1 = 500
nodesInH2 = 500
nodesInH3 = 500
nodesInH4 = 500


# Create first layer weights
layer_0_weights = tf.Variable(tf.random_normal([784, nodesInH1]))
layer_0_bias = tf.Variable(tf.random_normal([nodesInH1]))
layer_0 = tf.nn.relu(tf.add((tf.matmul(x_, layer_0_weights)), layer_0_bias))

# Create second layer weights
layer_1_weights = tf.Variable(tf.random_normal([nodesInH1, nodesInH2]))
layer_1_bias = tf.Variable(tf.random_normal([nodesInH2]))
layer_1 = tf.nn.relu(tf.add(tf.matmul(layer_0, layer_1_weights), layer_1_bias))

# Create third layer weights
layer_2_weights = tf.Variable(tf.random_normal([nodesInH2, nodesInH3]))
layer_2_bias = tf.Variable(tf.random_normal([nodesInH3]))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, layer_2_weights), layer_2_bias))

# Create fourth layer weights and bias
layer_3_weights = tf.Variable(tf.random_normal([nodesInH3, nodesInH4]))
layer_3_bias = tf.Variable(tf.random_normal([nodesInH4]))
layer_3 = tf.add(tf.matmul(layer_2, layer_3_weights), layer_3_bias)

# Create fifth layer weights and bias
layer_4_weights = tf.Variable(tf.random_normal([nodesInH4, 10]))
layer_4_bias = tf.Variable(tf.random_normal([10]))
layer_4 = tf.add(tf.matmul(layer_3, layer_4_weights), layer_4_bias)

prediction = layer_4
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y_))  # v1.0 changes
# optimizer with momentum and uses better algorithms
optimizer = tf.train.AdamOptimizer().minimize(cost)


N_EPOCHS = 10000
BATCH_SIZE = 200
errors = []
accuracies = []
accuraciesEpochNo = []
saver = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for i in range(N_EPOCHS):
    batch_X, batch_Y = mnist.train.next_batch(BATCH_SIZE)
    _, error = sess.run([optimizer, cost], feed_dict={x_: batch_X, y_: batch_Y})
    if(i% 100 ==0):
        print()
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("epoch", i, "error", error)
        accuracies.append(accuracy.eval({x_: mnist.test.images, y_: mnist.test.labels}))
        accuraciesEpochNo.append(i)

    errors.append(error)
save_path = saver.save(sess, "./tmp/question3/ANN/Q3annSave.ckpt")
print("Model saved in path: %s" % save_path)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('Final Accuracy:', accuracy.eval({x_: mnist.test.images, y_: mnist.test.labels}))



# plot error function
plt.plot(errors)
plt.title("Error function for session")
plt.xlabel("Epoch")
plt.ylabel("Error percentage")
plt.show()

# plot accuracy function
plt.plot(accuraciesEpochNo, accuracies)
plt.title("Accuracy function for session")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()



