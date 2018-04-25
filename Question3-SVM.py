import csv
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot=False)


x = []
y = []
x = mnist.train.images
y = mnist.train.labels


svm = SVC(C= 1, kernel='linear', gamma=0.8)
start = time.time()
print("fitting...")
svm.fit(x, y)
print("done")
end = time.time()
timeTaken = end - start;
print("time elapsed", timeTaken)

testX = mnist.test.images
testY = mnist.test.labels

predictions = svm.predict(testX)
numberCorrect = 0.0

for target, prediction in zip(testY,predictions):
    if target == prediction:
        numberCorrect += 1.0

accuracy = numberCorrect/float(len(testY))
print("Accuracy", accuracy)


for target, prediction in zip(testY, predictions):
    print("target",target,"prediction",prediction)

import pickle
pickle.dump(svm, open('./tmp/question3/SVM/Q3Dump.txt', 'wb'))
