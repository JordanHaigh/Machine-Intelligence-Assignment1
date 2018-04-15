import csv
from sklearn.svm import SVC
import scipy
from numpy import arange, round, meshgrid, resize
import matplotlib.pyplot as plt
import numpy as np

def read_dataset(filename="spirals/spiralsdataset.txt"):
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

svm = SVC(C=0.6, kernel="rbf", gamma=6)
yarray = np.ravel(y)



svm.fit(x, yarray)

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