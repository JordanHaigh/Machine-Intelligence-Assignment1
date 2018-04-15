import tensorflow as tf
from numpy import arange, round, meshgrid, math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder

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
# Entries with a '?' indicate a missing piece of data, and
# these entries are dropped from our dataset.

#go through column and find current num

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

#Validate and test datasets are same size


# The data that we need to add is the number of rows and columns present 
# in each of the training and testing CSV files. 
number_train_entries = df_train.shape[0]
number_train_features = df_train.shape[1] - 1

number_test_entries = df_test.shape[0]
number_test_features = df_test.shape[1] - 1

number_validate_entries = df_validate.shape[0]
number_validate_features = df_validate.shape[1]-1





############################SKIP FOR NOW####################################
# The data frames are written as a temporary CSV file, as we still
# need to modify the header row to include the number of rows and
# columns present in the training and testing CSV files.
df_train.to_csv('mushrooms/train_temp.csv', index=False)
df_test.to_csv('mushrooms/test_temp.csv', index=False)
df_validate.to_csv('mushrooms/validate_temp.csv', index=False)

# Append onto the header row the information about how many
# columns and rows are in each file as TensorFlow requires.
open("mushrooms/mushroom_train.csv", "w").write(str(number_train_entries) +
                                      "," + str(number_train_features) +
                                      "," + open("mushrooms/train_temp.csv").read())

open("mushrooms/mushroom_test.csv", "w").write(str(number_test_entries) +
                                     "," + str(number_test_features) +
                                     "," + open("mushrooms/test_temp.csv").read())




open("mushrooms/mushroom_validate.csv", "w").write(str(number_validate_entries) +
                                     "," + str(number_validate_features) +
                                     "," + open("mushrooms/validate_temp.csv").read())



##############################################################################
# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='mushrooms/mushroom_train.csv',
    target_dtype=np.int,
    features_dtype=np.int,
    target_column=0)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='mushrooms/mushroom_test.csv',
    target_dtype=np.int,
    features_dtype=np.int,
    target_column=0)


validate_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='mushrooms/mushroom_validate.csv',
    target_dtype=np.int,
    features_dtype=np.int,
    target_column=0)


#############################################################################
# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=116)] #CHANGE TO 116

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=2,
    model_dir="/tmp/mushroom_model")

################################################################################
# Define the test inputs
def get_test_inputs():
  x = tf.constant(test_set.data)
  y = tf.constant(test_set.target)

  return x, y

# Define the training inputs
def get_train_inputs():
  x = tf.constant(training_set.data)
  y = tf.constant(training_set.target)

  return x, y
##########################################################################
  # Fit model.
classifier.fit(input_fn=get_train_inputs, steps=2000)

##########################################################################
# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                     steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


##########################################################################
# Test on two mushroom samples.

# Test on two mushroom samples.
predictions = list(classifier.predict(df_validate.iloc[:,1:].values))

print("New Samples, Class Predictions:    {}\n"
      .format(predictions))


