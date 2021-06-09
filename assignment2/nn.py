import numpy as np
import math as m

class NeuralNetwork:

    def __init__(self, lr, batchSize, epochs):
		# Method to initialize a Neural Network Object
		# Parameters
		# lr - learning rate
		# batchSize - Mini batch size
		# epochs - Number of epochs for training
        self.lr = lr
        self.batchSize = batchSize
        self.epochs = epochs
        self.layers = []
        self.activationslist = []
    
    def convertzero_one(self,values):
        return np.where(values < 0.5, 1e-8, 1)
        
    def addLayer(self, layer):
        # Method to add layers to the Neural Network
        self.layers.append(layer)

    def train(self, trainX, trainY, validX=None, validY=None):

         #print(trainX)
         for i in range(self.epochs):
             #print("loop ",i)
             XY = np.concatenate((trainX,trainY),axis=1)
             traincolsize = trainX.shape[1]
             np.random.shuffle(XY)

             trainX = XY[:,:traincolsize]
             trainY = XY[:,traincolsize:XY.shape[1]]
             trainX = np.squeeze(trainX)
             trainY = np.squeeze(trainY)
             miniTrainX = trainX[:self.batchSize]
             miniTrainY = trainY[:self.batchSize]
             #miniTrainX += np.full_like(miniTrainX,1e-8)
             #print("trainx mini",miniTrainX)
             #np.random.shuffle(randomize)
             #self.lr = 1/m.sqrt(self.epochs)
             predY = self.fullForwardPass(miniTrainX)
             
             np.where(predY<1e-10,0.9999,predY)
             #print("predY",predY)
             #print("train accuracy",self.computeAccuracy(miniTrainY,predY))
             c_loss = self.crossEntropyLoss(miniTrainY,predY) #/trainX.shape[0]
             
             #if(i%10 ==0):
             #print("loss :",c_loss)
             c_lossder = self.crossEntropyDelta(miniTrainY,predY) #/trainX.shape[0]
             
             #print("c loss derivative",c_lossder)
             self.fullBackwardPass(predY,c_lossder)         
        # print("Valid",self.validate(validX,validY))      
        
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# Feel free to print accuracy at different points using the validate() or computerAccuracy() functions of this class
		###############################################
		# TASK 2c (Marks 0) - YOUR CODE HERE
         #raise NotImplementedError
		###############################################
		
    def crossEntropyLoss(self, Y, predictions):

        predictions[predictions == 0] = 1e-8
        return -np.sum(Y*np.log(predictions)) 
    

    def crossEntropyDelta(self, Y, predictions):
         
         with np.errstate(divide='ignore', invalid='ignore'):
             crossdel = np.true_divide(Y,predictions )
             crossdel[ ~ np.isfinite( crossdel )] = 0  
         return -crossdel

         
    def computeAccuracy(self, Y, predictions):
		# Returns the accuracy given the true labels Y and final output of the model
        correct = 0
        for i in range(len(Y)):
            if np.argmax(Y[i]) == np.argmax(predictions[i]):
                correct += 1
        accuracy = (float(correct) / len(Y)) * 100
        return accuracy

    def validate(self, validX, validY):
		# Input 
		# validX : Validation Input Data
		# validY : Validation Labels
		# Returns the predictions and validation accuracy evaluated over the current neural network model
        valActivations = self.predict(validX)
        pred = np.argmax(valActivations, axis=1)
        if validY is not None:
            valAcc = self.computeAccuracy(validY, valActivations)
            return pred, valAcc
        else:
            return pred, None

    def predict(self, X):
        activations = X
        for l in self.layers:
            activations = l.forwardpass(activations)
        return activations

		# Input
		# X : Current Batch of Input Data as an nparray
		# Output
		# Returns the predictions made by the model (which are the activations output by the last layer)
		# Note: Activations at the first layer(input layer) is X itself		
    def fullForwardPass(self, trainX):
        #layer_count = len(self.layers)
        i = 0
        curdata = np.array(trainX)
        self.activationslist = []
        self.activationslist.append(curdata)
        for layer in self.layers:  
            #print("This is forward pass number ",i)
            next_pass_data = layer.forwardpass((curdata))
            curdata = np.array(next_pass_data, copy=True)
            #print("current data",curdata)
            self.activationslist.append((curdata))
            i +=1
        #print (self.activationslist)
        return curdata
    
    def fullBackwardPass(self,predictedY,c_lossder):
        delta = c_lossder
        preact = -2
        #self.layers.reverse()
        for layer in reversed(self.layers):
            delta = layer.backwardpass(self.activationslist[preact],delta)
            #layer.updateWeights(self.lr)
            preact -= 1
        for layer in reversed(self.layers):
            layer.updateWeights(self.lr)
        #self.layers.reverse()    





class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes, activation):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
         self.in_nodes = in_nodes
         self.out_nodes = out_nodes
         self.activation = activation
         # Stores a quantity that is computed in the forward pass but actually used in the backward pass. Try to identify
		# this quantity to avoid recomputing it in the backward pass and hence, speed up computation
         self.data = None
         #np.random.seed(50)
		# Create np arrays of appropriate sizes for weights and biases and initialise them as you see fit
		###############################################
		# TASK 1a (Marks 0) - YOUR CODE HERE
		#raise NotImplementedError
         #self.weights = np.random.randn(in_nodes,out_nodes)*0.1
         #self.biases = np.random.randn(1,out_nodes)*0.1
         #self.weights = np.random.randint(100, size=(in_nodes, out_nodes))*0.0001
         #self.biases = np.random.randint(100, size=(1, out_nodes))*0.01
         #rng = np.random.RandomState(1234)

         #a = 1/ in_nodes
#         self.weights= np.array(rng.uniform(  # initialize W uniformly
#                low=-a,
#                high=a,
#                size=(in_nodes, out_nodes)))*0.1

         #self.biases = np.zeros([1,out_nodes],dtype=float)
         self.weights = np.random.normal(0,1,in_nodes*out_nodes).reshape(in_nodes,out_nodes)
         self.biases = np.random.normal(0,1,out_nodes).reshape(1,out_nodes)
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary
		
		# Store the gradients with respect to the weights and biases in these variables during the backward pass
         self.weightsGrad = None
         self.biasesGrad = None

	def relu_of_X(self, X):
         return np.maximum(X,0)


	def gradient_relu_of_X(self, X, delta):
         gradient = np.array(delta,copy =True)
         gradient[X<=0]=0
         gradient[X>0]=1
         return gradient*delta

	def softmax_of_X(self, X):

         size = X.shape[0]
         res = []

         for i in range(size):
             expstable = np.exp(X[i] - X[i].max())
             temp = expstable/np.float(np.sum(expstable))
             res.append(temp)
         #print("softmax of X",np.array(res))
         
         return np.array(res)

	def gradient_softmax_of_X(self, X, delta):
         size = X.shape[0]
         jacobian = []
         for i in range(size):
             temp = X[i].reshape(-1,1)
             jac = np.diagflat(temp) - np.dot(temp, temp.T)
             #if(i==1):
              #   print("jac",jac)
             #jac = np.dot(delta[i],jac)
             jac = np.dot(jac,delta[i])
             jacobian.append(jac)
             
         #jacobian = np.squeeze(jacobian) 
         #print("jac shape",np.array(jacobian).shape)
         #print("Jacobian",np.array(jacobian))
         return np.array(jacobian)
         #return (np.dot(delta,jacobian))

	def forwardpass(self, X):
         #print("input in forward pass",X)
         #print("self weights",self.weights)
         #print("self biases",self.biases)
         output = np.dot(X,self.weights)  + self.biases
         #print("output in Forward before activation",output)
         if self.activation == 'relu':
             self.data = self.relu_of_X(output)
             #print("output in Forward pass after activation",self.data)
             return self.data
         elif self.activation == 'softmax':
             self.data = self.softmax_of_X(output)
             #print("output in Forward pass after activation",self.data)
             return self.data
         else:
             print("ERROR: Incorrect activation specified: " + self.activation)
             exit()

	def backwardpass(self, activation_prev, delta):
         #print("Backward pass starts here")
         #print("act_prev",activation_prev)
         #print("delta",delta)
         #print("self.data",self.data)
         if self.activation == 'relu':
             inp_delta = self.gradient_relu_of_X(self.data, delta)
             #print("inp_delta relu",inp_delta)
         elif self.activation == 'softmax':
             inp_delta = self.gradient_softmax_of_X(self.data, delta)
             #print("inp_delta softmax",inp_delta)
         else:
             print("ERROR: Incorrect activation specified: " + self.activation)
             exit()
         #print("previous activation",activation_prev.shape)
         #print("ghotala wala input delta",inp_delta)
         #print("inp_Delta",inp_delta)
         #print("activation_prev",activation_prev)
        
         self.weightsGrad = (np.dot(inp_delta.T,activation_prev).T) /inp_delta.shape[0]
         #print(self.weightsGrad.shape)
         self.biasesGrad = (np.sum( (inp_delta), axis=0, keepdims=False)) / inp_delta.shape[0]
         output_der = np.dot(inp_delta,self.weights.T)
         return output_der
		# Input
		# activation_prev : Output from next layer/input | shape: batchSize x self.out_nodes]
		# delta : del_Error/ del_activation_curr | shape: self.out_nodes
		# Output
		# new_delta : del_Error/ del_activation_prev | shape: self.in_nodes
		# You may need to write different code for different activation layers

		# Just compute and store the gradients here - do not make the actual updates
		###############################################
		# TASK 1g (Marks 6) - YOUR CODE HERE

		###############################################
		
	def updateWeights(self, lr):

         self.weights -=  lr*self.weightsGrad
         self.biases  -= lr*self.biasesGrad

'''
x =np.array([[0.01275478, 0.03467109, 0.25618664 ,0.69638749],[0.01275478, 0.03467109 ,0.25618664, 0.69638749]])
de = np.array([[0, 1, 0, 0],[0, 1, 0, 0]])
nn1 = NeuralNetwork(0.0, 1, 1)
nn1.addLayer(FullyConnectedLayer(4,4,'softmax'))
print(FullyConnectedLayer(4,4,'softmax').gradient_softmax_of_X(x,de))
'''