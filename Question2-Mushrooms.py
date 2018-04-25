import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import arange, round, meshgrid, math


from sklearn.model_selection import train_test_split

headers = ["Edible/Poisonous","cap-shape","cap-surface","cap-color","bruises",
              "odor","gill-attachment","gill-spacing","gill-size",
              "gill-color","stalk-shape","stalk-root","stalk-surface-above-ring",
              "stalk-surface-below-ring","stalk-color-above-ring",
              "stalk-color-below-ring","veil-type","veil-color",
              "ring-number","ring-type","spore-print-color",
              "population","habitat"]


def csv_to_dataframe(filename="mushrooms/agaricus-lepiota.data.txt"):
#def csv_to_dataframe(filename="mushrooms/newmush.txt"):
    df = pd.read_csv(filename, header=None,names=headers)

    return df


df = csv_to_dataframe()

#########################
# Entries with a '?' indicate a missing piece of data

#Use mean of column to replace '?'
#Modified from https://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series(
                [X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)



df.replace('?', np.nan, inplace=True)
df = DataFrameImputer().fit_transform(df)
##########

#data cleanse modified from 
#http://vprusso.github.io/blog/2017/tensor-flow-categorical-data/#download-and-clean-the-mushroom-data-from-the-uci-repository

#Replace p or e with 0 or 1 depending on poisonous or edible
df['Edible/Poisonous'].replace('p', 0, inplace=True)
df['Edible/Poisonous'].replace('e', 1, inplace=True)


# Since we are dealing with non-numeric feature (categorical) data, 
# we need to replace these with numerical equivalents. 
cols_to_transform = headers[1:]
df = pd.get_dummies(df, columns=cols_to_transform)






#split data for training and testing
#test data is 20 percent of entire data
df_train, df_test = train_test_split(df, test_size=0.2)
df_test, df_validate = train_test_split(df_test, test_size=0.5)

x = df_train.iloc[:,1:]
y = df_train.iloc[:,0:1]


#Validate and test datasets are same size


# The data that we need to add is the number of rows and columns present 
# in each of the training and testing CSV files. 
number_train_entries = df_train.shape[0]
number_train_features = df_train.shape[1] - 1

number_test_entries = df_test.shape[0]
number_test_features = df_test.shape[1] - 1

number_validate_entries = df_validate.shape[0]
number_validate_features = df_validate.shape[1]-1


##########################################################################################



# Create the model
x_ = tf.placeholder(tf.float32, [None, 116])
y_ = tf.placeholder(tf.float32, [None, 1])

nodesInH1 = 10
nodesInH2 = 20
nodesInH3 = 10

# Create first layer weights
layer_0_weights = tf.Variable(tf.random_normal([116, nodesInH1]))
layer_0_bias = tf.Variable(tf.random_normal([nodesInH1]))
layer_0 = tf.nn.sigmoid(tf.add((tf.matmul(x_, layer_0_weights)), layer_0_bias))

# Create second layer weights
layer_1_weights = tf.Variable(tf.random_normal([nodesInH1, nodesInH2]))
layer_1_bias = tf.Variable(tf.random_normal([nodesInH2]))
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_0, layer_1_weights), layer_1_bias))

layer_mid_weights = tf.Variable(tf.random_normal([nodesInH2, nodesInH3]))
layer_mid_bias = tf.Variable(tf.random_normal([nodesInH3]))
layer_mid = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, layer_mid_weights), layer_mid_bias))

# Create third layer weights
layer_2_weights = tf.Variable(tf.random_normal([nodesInH3, 1]))
layer_2_bias = tf.Variable(tf.random_normal([1]))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_mid, layer_2_weights), layer_2_bias))

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
    print(i, 'error count: ', error)
    if error < 0.015:
        print("reached less than 0.01 > epoch: ",i, 'error count: ', error)
        break

save_path = saver.save(sess, "./tmp/question2/model.ckpt")
print("Model saved in path: %s" % save_path)

tf.reset_default_graph()

saver.restore(sess, "./tmp/question2/model.ckpt")
print("Model restored.")

plt.plot(errors)
plt.title("Error function for session")
plt.xlabel("Epoch")
plt.ylabel("Error percentage")
plt.show()

classifications = round(sess.run(layer_2, feed_dict={x_: x}))


correct = 0
incorrect = 0
for coord, target, prediction in zip(x.values.tolist(),y.values.tolist(),classifications):
    if(round(prediction[0]) == target[0]):
        correct +=1
    else:
        incorrect +=1
barx = ["Correct","Incorrect"]
bary = [correct,incorrect]
width = 1/1.5
plt.bar(barx, bary, width, color="blue")

plt.show()



###########################################################################################
###########################################################################################
#
## The data frames are written as a temporary CSV file
## still need to modify the header row to include the number of rows and
## columns in the training, testing and validation files.
#df_train.to_csv('mushrooms/train_temp.csv', index=False)
#df_test.to_csv('mushrooms/test_temp.csv', index=False)
#df_validate.to_csv('mushrooms/validate_temp.csv', index=False)
#
## Append onto the header row the information about how many
## columns and rows are in each file for tensorflow dataset.
#open("mushrooms/mushroom_train.csv", "w").write(str(number_train_entries) +
#                                      "," + str(number_train_features) +
#                                      "," + open("mushrooms/train_temp.csv").read())
#
#open("mushrooms/mushroom_test.csv", "w").write(str(number_test_entries) +
#                                     "," + str(number_test_features) +
#                                     "," + open("mushrooms/test_temp.csv").read())
#
#
#
#
#open("mushrooms/mushroom_validate.csv", "w").write(str(number_validate_entries) +
#                                     "," + str(number_validate_features) +
#                                     "," + open("mushrooms/validate_temp.csv").read())
#
#
#
## Load datasets.
#training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#    filename='mushrooms/mushroom_train.csv',
#    target_dtype=np.int,
#    features_dtype=np.int,
#    target_column=0)
#
#test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#    filename='mushrooms/mushroom_test.csv',
#    target_dtype=np.int,
#    features_dtype=np.int,
#    target_column=0)
#
#
#validate_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#    filename='mushrooms/mushroom_validate.csv',
#    target_dtype=np.int,
#    features_dtype=np.int,
#    target_column=0)
#
#
## Specify that all features have real-value data
#feature_columns = [tf.contrib.layers.real_valued_column("", dimension=116)]
##Modified from https://stackoverflow.com/a/42295872
## Build 3 layer DNN with 10, 20, 10 units respectively.
#classifier = tf.contrib.learn.DNNClassifier(
#    feature_columns=feature_columns,
#    hidden_units=[10, 20, 10],
#    n_classes=2,
#    model_dir="/tmp/mushroom_model")
#
##Tensorflow comes with its own Dataset type where you are able to access data and target sections
#
## Functions useful for input functions to classifier.fit
#def get_train_inputs():
#  x = tf.constant(training_set.data)
#  y = tf.constant(training_set.target)
#
#  return x, y
#
#def get_test_inputs():
#  x = tf.constant(test_set.data)
#  y = tf.constant(test_set.target)
#
#  return x, y
#
#
## Fit model and run
#classifier.fit(input_fn=get_train_inputs, steps=2000)
#
## Evaluate accuracy of fit function
#accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
#                                     steps=1)["accuracy"]
#
#print("\nTest Accuracy:", accuracy_score)
#
##Now try and run NN on validation set to confirm that it will work
#predictions = list(classifier.predict(df_validate.iloc[:,1:].values))
#
#print("New Samples, Class Predictions:    {}\n"
#      .format(predictions))
#


#
#
##redo predictions for bar graph
#predictions = list(classifier.predict(df.iloc[:,1:].values))
#
#edible = 0
#poisonous = 0
#for i in predictions:
#    if(i == 1):
#        edible +=1
#    else:
#        poisonous +=1
#
#bary = [edible, poisonous]
#barx = ["Edible","Poisonous"]
#width=1/2
#plt.title("Classification of edible and poisonous mushrooms")
#plt.xlabel("Classification")
#plt.ylabel("Count")
#plt.bar(barx, bary, width, color="blue")
#


