import csv
from sklearn.svm import SVC
import scipy
from numpy import arange, round, meshgrid, resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def csv_to_dataframe(filename="spirals/2SpiralsRotated90.txt"):
    df = pd.read_csv(filename, header=None, names = ["x","y","spiralid"])
    x = df.iloc[:,0:2]
    y = df.iloc[:,2:3]
    return df, x, y


df, x, y = csv_to_dataframe()

labels = {1: 'spiral1', 2: 'spiral2', 3: 'spiral3', 4: 'spiral4'}
colors = {1: 'r', 2:'g', 3:'b', 4:'yellow'}
plt.figure(1)



svm = SVC(C=0.6, kernel="rbf", gamma=6)
svm.fit(x, y)

# Visualise activations
activation_range = arange(-6,6,0.1) # interval of [-6,6) with step size 0.1
coordinates = [(x,y) for x in activation_range for y in activation_range]
classifications = svm.predict(coordinates)
meshx, meshy = meshgrid(activation_range, activation_range)
plt.scatter(meshx, meshy, c=['b' if x > 0 else 'y' for x in classifications])

plt.title("Classification of 2 Spirals")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



##########################################

#classifications = round(sess.run(layer_2, feed_dict={x_: x}))
classifications = svm.predict(x)

plt.title("Predictions -  Spiral 1: Red, Spiral 2; Blue")
plt.xlabel("X")
plt.ylabel("Y")
for coord, cls in zip(x,classifications):
    #print(coord[0], coord[1], cls)
    if(cls == 0):
        plt.scatter(coord[0],coord[1],c = "red" )
    else:
        plt.scatter(coord[0],coord[1],c = "blue" )
plt.show()




plt.title("Correct Predictions: Green, Incorrect Predictions: Red")
plt.xlabel("X")
plt.ylabel("Y")
for coord, target, prediction in zip(x,yarray,classifications):
    if(round(prediction) == target):
        plt.scatter(coord[0], coord[1],c = "green" )
    else:
        plt.scatter(coord[0], coord[1], c = "red")

plt.show()