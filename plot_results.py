import matplotlib.pyplot as plt


abs_loss = 3
mse_loss = 4
accuracy = 5

data = [
# TYPE(1,3,10+knn), WEEKEND, NORMALIZED, MA_ERROR, MSE ROOT, ACCURACY, LOG
    [1, False, False, 434.443, 1031.185, 0.255, False],     #0
    [1, False, True, 434.443, 1031.185, 0.255, False],
    [1, True, False, 438.320, 997.219, 0.277, False],
    [1, True, True, 438.320, 997.219, 0.255, False],
    [17, False, False, 267.048, 776.491, 0.476, False],     #4
    [17, False, True, 256.766, 754.178, 0.490, False],
    [17, True, False, 267.031, 776.490, 0.476, False],
    [15, True, True, 238.595, 745.434, 0.520, False],       #7
    [17, True, True, 234.386, 727.947, 0.527, False],       
    [20, True, True, 231.348, 733.609, 0.533, False],
    [23, True, True, 229.179, 731.009, 0.539, False],
    [26, True, True, 229.213, 733.701, 0.540, False],       #11

    [1, True, False, 0.364, 0.637, 0.291, True],            #12
    [17, True, False, 0.229, 0.481, 0.503, True],
    [15, True, True, 0.232, 0.484, 0.499, True],            #14
    [17, True, True, 0.226, 0.471, 0.504, True],
    [19, True, True, 0.224, 0.467, 0.507, True],
    [20, True, True, 0.221, 0.461, 0.505, True],
    [23, True, True, 0.223, 0.466, 0.504, True],
]

data_comb = [
# threshold, knn, accuracy
    [1500 ,9, 0.522],
    [1000 ,9, 0.523],
    [900 ,9, 0.523],
    [700 ,9, 0.523],
    [500 ,9, 0.524],
    [300 ,9, 0.526],
    [200 ,9, 0.528],
    [100 ,9, 0.530],
    [50 ,9, 0.532],
    [20 ,9, 0.53281],
    [0 ,9, 0.5332],
]

comb_lower = 0.5332
comb_higher = 0.5098


def addlabels(coordx,y,ax=plt):
    for i in range(len(coordx)):
        ax.text(coordx[i]-0.15,y[i]/2,round(y[i],2))

def plot1():
    X1 = ["Linear without weekdays", "Linear with weekdays"]
    Y1 = [data[0][abs_loss],data[2][abs_loss]]
    X2 = ["Knn without weekdays", "Knn with weekdays"]
    Y2 = [data[4][abs_loss],data[6][abs_loss]]
    plt.title("")
    plt.ylabel("Mean absolute error")


    addlabels([0,1],Y1)
    addlabels([2,3],Y2)
    plt.bar(X1,Y1)
    plt.bar(X2,Y2)


def plot3():
    X2 = ["knn - non normalized", "knn - normalized"]
    Y1 = [data[6][mse_loss],data[7][mse_loss]]
    Y2 = [data[6][accuracy],data[7][accuracy]]

    fig, ax = plt.subplots(1,2)
    
    addlabels([0,1],Y1,ax[0])
    addlabels([0,1],Y2,ax[1])
    #plt.bar(X1,Y1)
    ax[0].bar(X2,Y1)
    ax[0].set_ylabel("MSE loss")
    ax[1].bar(X2,Y2)
    ax[1].set_ylabel("Accuracy")

def plot2():
    X = [x[0] for x in data_comb]
    print(X)
    Y = [x[2] for x in data_comb]
    plt.title("Combined model accuracy using different thresholds")
    plt.ylabel("Accuracy")
    plt.xlabel("Threshold")
    plt.plot(X,Y,label='Combined model accuracy')
    plt.plot([0,max(X)],[comb_lower,comb_lower],label='Normal model accuracy')
    plt.plot([0,max(X)],[comb_higher,comb_higher],label='Log model accuracy')
    plt.legend()


def plot4():
    X = [data[i][0]-10 for i in range(14,19)]
    Y1 = [data[i][mse_loss] for i in range(14,19)]
    Y2 = [data[i][accuracy] for i in range(14,19)]

    fig, ax = plt.subplots(1,2)

    ax[0].plot(X,Y1)
    ax[0].set_ylabel("MSE loss")
    ax[0].set_xlabel("Number of clusters")
    ax[1].plot(X,Y2)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Number of clusters")

plot2()
#plot2()
plt.show()