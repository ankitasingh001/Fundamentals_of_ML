import numpy as np
import nn
import csv
import pickle
import matplotlib.pyplot as plt

def taskXor():
    XTrain, YTrain, XVal, YVal, XTest, YTest = loadXor()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
    nn1 = nn.NeuralNetwork(0.1, 500, 500)
    nn1.addLayer(nn.FullyConnectedLayer(2,5,'relu'))
    nn1.addLayer(nn.FullyConnectedLayer(5,3,'relu'))
    nn1.addLayer(nn.FullyConnectedLayer(3,2,'relu'))
    nn1.addLayer(nn.FullyConnectedLayer(2,2,'softmax'))

    nn1.train(XTrain, YTrain, XVal, YVal)

    pred, acc = nn1.validate(XTest, YTest)
    with open("predictionsXor.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prediction"])
        for i, p in enumerate(pred):
            writer.writerow([i, p])
    print('Test Accuracy',acc)
    return nn1

def preprocessMnist(X):
    
    scaletrain = np.ones_like(X)
    for i in range(X.shape[0]):
        scaletrain[i] = (np.asfarray(np.array(X[i])) * 0.999)/255 + 0.001
    #print(scaletrain.shape)
    
    return scaletrain

	# Perform any data preprocessing that you wish to do here
	# Input: A 2-d numpy array containing an entire train, val or test split | Shape: n x 28*28
	# Output: A 2-d numpy array of the same shape as the input (If the size is changed, you will get downstream errors)
	###############################################
	# TASK 3c (Marks 0) - YOUR CODE HERE
	#raise NotImplementedError
	###############################################

def taskMnist():
    XTrain, YTrain, XVal, YVal, XTest, _ = loadMnist()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
    nn1 = nn.NeuralNetwork(0.4, 500, 500)
    nn1.addLayer(nn.FullyConnectedLayer(784,500,'relu'))
    #nn1.addLayer(nn.FullyConnectedLayer(300,100,'relu'))
    #nn1.addLayer(nn.FullyConnctedLayer(800,500,'relu'))
    #nn1.addLayer(nn.FullyConnectedLayer(500,100,'relu'))
    nn1.addLayer(nn.FullyConnectedLayer(500,10,'softmax'))

	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 3b (Marks 13) - YOUR CODE HERE
	#raise NotImplementedError
	###############################################
    nn1.train(XTrain, YTrain, XVal, YVal)
    pred, _ = nn1.validate(XTest, None)
    with open("predictionsMnist.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prediction"])
        for i, p in enumerate(pred):
            writer.writerow([i, p])
    return nn1

################################# UTILITY FUNCTIONS ############################################
def oneHotEncodeY(Y, nb_classes):
	# Calculates one-hot encoding for a given list of labels
	# Input :- Y : An integer or a list of labels
	# Output :- Coreesponding one hot encoded vector or the list of one-hot encoded vectors
     #print(Y)
     return (np.eye(nb_classes)[Y]).astype(int)

def loadXor():
	# This is a toy dataset with 10k points and 2 labels.
	# The output can represented as the XOR of the input as described in the problem statement
	# There are 7k training points, 1k validation points and 2k test points
    train = pickle.load(open("data/xor/train.pkl", 'rb'))
    test = pickle.load(open("data/xor/test.pkl", 'rb'))
    testX, testY = np.array(test[0]), np.array(oneHotEncodeY(test[1],2))
    trainX, trainY = np.array(train[0][:7000]), np.array(oneHotEncodeY(train[1][:7000],2))
    valX, valY = np.array(train[0][7000:]), np.array(oneHotEncodeY(train[1][7000:],2))

    return trainX, trainY, valX, valY, testX, testY

def loadMnist():
	# MNIST dataset has 50k train, 10k val, 10k test
	# The test labels have not been provided for this task
    train = pickle.load(open("data/mnist/train.pkl", 'rb'))
    test = pickle.load(open("data/mnist/test.pkl", 'rb'))
    #train = np.array(train)
    #test = np.array(test)
    #print(np.array(train[0][:50000]).shape)
    #print(test[0][1].shape)
    #img = np.array(train[0][1]).reshape((28,28))
    #plt.imshow(img, cmap="Greys")
    #plt.show()
    
    testX = preprocessMnist(np.array(test[0]))
    #print(train[0][0])
    #print(test[0][0])
    testY = None # For MNIST the test labels have not been provided
    
    trainX, trainY = preprocessMnist(np.array(train[0][:50000])), np.array(oneHotEncodeY(train[1][:50000],10))
    valX, valY = preprocessMnist(np.array(train[0][50000:])), np.array(oneHotEncodeY(train[1][50000:],10))
    
    return trainX, trainY, valX, valY, testX, testY
    
#################################################################################################

if __name__ == "__main__":
	np.random.seed(7)
	taskXor()
	taskMnist()

