import csv
from sklearn.svm import SVC
import scipy
from numpy import arange, round, meshgrid, resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def csv_to_dataframe(filename="spirals/chang_spiral.csv"):
    df = pd.read_csv(filename, header=None, names = ["x","y","spiralid"])
    x = df.iloc[:,0:2]
    y = df.iloc[:,2:3]
    return df, x, y


df, x, y = csv_to_dataframe()

labels = {1: 'spiral1', 2: 'spiral2', 3: 'spiral3', 4: 'spiral4'}
colors = {1: 'r', 2:'g', 3:'b', 4:'yellow'}

#It works!
for index, rows in df.iterrows():
    xcoord = rows["x"]
    ycoord = rows["y"]
    spiralid = rows["spiralid"]
    a = int(spiralid)
    color = colors[a]
    
    plt.scatter(xcoord,ycoord, c=color)
        
#






svm = SVC(C=0.6, kernel="rbf", gamma=6)

yarray = np.ravel(y)
svm.fit(x, yarray)

#Visualise activations
activation_range = arange(0,35,0.1) # interval of [-6,6) with step size 0.1
coordinates = [(x,y) for x in activation_range for y in activation_range]
classifications = svm.predict(coordinates)
meshx, meshy = meshgrid(activation_range, activation_range)
plt.scatter(meshx, meshy, c=['r' if x ==3 else ('g' if x == 2 else 'b') for x in classifications])

plt.title("Classification of 3 Spirals")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



##########################################

#classifications = round(sess.run(layer_2, feed_dict={x_: x}))
classifications = svm.predict(x)

plt.title("Predictions -  Spiral 1: Red, Spiral 2; Blue, Spiral 3: Green")
plt.xlabel("X")
plt.ylabel("Y")
#It works!
for index, rows in df.iterrows():
    xcoord = rows["x"]
    ycoord = rows["y"]
    spiralid = rows["spiralid"]
    a = int(spiralid)
    color = colors[a]
    
    plt.scatter(xcoord,ycoord, c=color)
     




plt.title("Correct Predictions: Green, Incorrect Predictions: Red")
plt.xlabel("X")
plt.ylabel("Y")
for coord, target, prediction in zip(x.values.tolist(),y.values.tolist(),classifications):
    if(round(prediction) == target):
        plt.scatter(coord[0], coord[1],c = "green" )
    else:
        plt.scatter(coord[0], coord[1], c = "red")

plt.show()