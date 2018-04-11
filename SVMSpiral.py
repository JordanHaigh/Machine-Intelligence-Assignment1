import csv
from sklearn.svm import SVC
from numpy import arange, round, meshgrid, resize
import matplotlib.pyplot as plt


def read_two_spiral_file(filename="../datasets/spiralsdataset.csv"):
    x = []
    y = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            x.append(list(map(float, row[:-1])))
            y.append([int(row[-1])])

    return x, y


x, y = read_two_spiral_file()


svm = SVC(C=0.1, kernel='rbf', gamma=0.1)


svm.fit(x, y)

# Visualise activations
activation_range = arange(-6,6,0.1) # interval of [-6,6) with step size 0.1
coordinates = [(x,y) for x in activation_range for y in activation_range]
classifications = svm.predict(coordinates)
x, y = meshgrid(activation_range, activation_range)
plt.scatter(x, y, c=['b' if x > 0 else 'y' for x in classifications])
plt.show()