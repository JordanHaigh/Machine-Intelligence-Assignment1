import csv
from sklearn.svm import SVC
import scipy
from numpy import arange, round, meshgrid, resize
import matplotlib.pyplot as plt


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




svm = SVC(C=0.1, kernel='rbf', gamma=0.1)


svm.fit(x, y)

# Visualise activations
activation_range = arange(-6,6,0.1) # interval of [-6,6) with step size 0.1
coordinates = [(x,y) for x in activation_range for y in activation_range]
classifications = svm.predict(coordinates)
x, y = meshgrid(activation_range, activation_range)
plt.scatter(x, y, c=['b' if x > 0 else 'y' for x in classifications])
plt.show()