# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:45:39 2019

@author: chikki
"""

import numpy as np
import argparse
import csv
import datetime
import math
import matplotlib.pyplot as plt

# One hot encoding for dates            
def splitdate(n):
    datelist= []
    temp=[]
    date = []
    rows, cols = (len(n), 0) 
    temp = [[0]*cols]*rows 
    for i in n:
        k = i.split('-')
        datelist.extend(k)
    date=np.reshape(np.array(datelist),(len(n),3))
    date = date.astype(int)
    temp= np.append(temp,zerooneencode(date[:,2],31),axis=1)
    temp= np.append(temp,zerooneencode(date[:,1],12),axis=1)
    npyear = np.vectorize(getyear)
    year = npyear((date[:,0]))
    year =year.reshape(year.shape[0],-1)
    temp = np.append(temp,year,axis=1)
    return temp


# code for one hot encoding categorical attributes       
def zerooneencode(n,uniqlen):
    n = n.astype(int)
    row = len(n)
    col = uniqlen+1
    letter = np.zeros((row,col))
    for i in range(len(n)):
        letter[i,n[i]] = 1
    return letter

#map weekdays to integer values        
def getweeknum(weekday):
    d= {}    
    d = { "Monday" :0 , "Tuesday" :1 , "Wednesday" :2 ,"Thursday": 3, "Friday" :4,
    "Saturday":5, "Sunday":6 }
    return d[weekday]
    
#maps year to binary values
def getyear(n):
    d= {} 
    d= {2011 :0 , 2012 :1}
    return d[n]

#Correct noise in wind speed and humidity    
def correctnoisyspeed(n):

    b = np.where(n=='3',1,n)
    return b

def mean_squared_loss(xdata, ydata, weights):

    num = len(ydata)
    predicted_data = np.dot(xdata,weights)
    error = (predicted_data - ydata)**2
    return (1.0/(num) * error.sum())


def mean_squared_gradient(xdata, ydata, weights):
    
    num = len(ydata)
    predicted_data = np.dot(xdata,weights)
    error_der = 2*(predicted_data - ydata)
    gradient = np.dot(xdata.T,  error_der)
    #gradient /= num
    return np.true_divide(gradient, num)
	

def mean_absolute_loss(xdata, ydata, weights):
    
    num = len(ydata)
    predicted_data = np.dot(xdata,weights)
    error = abs(predicted_data - ydata)
    return (1.0/(num) * error.sum())


def mean_absolute_gradient(xdata, ydata, weights):
    num = len(ydata)
    predicted_data = np.dot(xdata,weights)
    value = predicted_data - ydata
    gradient = np.dot(xdata.T, np.sign(value))
    gradient = gradient/num
    return gradient


def mean_log_cosh_loss(xdata, ydata, weights):
    num = len(ydata)
    predicted_data = np.dot(xdata,weights)
    var = np.minimum(np.ones(num)*100,np.abs(predicted_data - ydata))
    value = np.log(np.cosh(var))
    return (1.0/(num)*np.sum(value))


def mean_log_cosh_gradient(xdata, ydata, weights):
    num = len(ydata)
    predicted_data = np.dot(xdata,weights)
    error_der =np.tanh(predicted_data - ydata)
    gradient = np.dot(xdata.T,  error_der)
    gradient /= num
    return gradient

def root_mean_squared_loss(xdata, ydata, weights):
    
    num = len(ydata)
    predicted_data = np.dot(xdata,weights)
    error = (predicted_data - ydata)**2
    return np.sqrt(1.0/(num) * error.sum())

def root_mean_squared_gradient(xdata, ydata, weights):
    num = len(ydata)
    gradient = mean_squared_gradient(xdata,ydata,weights)*num;
    predicted_data = np.dot(xdata,weights)
    error = (predicted_data - ydata)**2
    gradient = gradient/(2*math.sqrt(error.sum()*(num)))
    return gradient


class LinearRegressor:
    def __init__(self,dims):
		
           self.weights = np.ones(dims).T

    def train(self, xtrain, ytrain, loss_function, gradient_function, epoch=10000, lr=0.0001):
        
        loss = loss_function(xtrain,ytrain,self.weights)
        lossprev = math.inf
        for i in range(epoch):
             gradient = gradient_function(xtrain,ytrain,self.weights)
             self.weights -= np.dot(gradient,lr)
             loss = loss_function(xtrain,ytrain,self.weights)
             if(lossprev<loss):
                 break
#             if(i%100 ==0):
#                 print (loss)
             lossprev =loss
            
        return self.weights        

    def predict(self, xtest):
           
           ytest = np.dot(xtest,self.weights)
           #ytest = np.power(10,ytest)

           yvalues=np.around(ytest, decimals=0)
           yvalues[yvalues < 0] = 1
           print("instance (id),count")
           id= np.arange(len(yvalues))
           #output = np.concatenate((np.arange(len(yvalues)).reshape(-1,1) ,yvalues.reshape(-1,1)),axis=1)
           #print(output)
           for i in range(len(yvalues)):
               print(id[i],",",yvalues[i])
           np.savetxt("pred.csv",np.concatenate((np.arange(len(yvalues)).reshape(-1,1) ,yvalues.reshape(-1,1)),axis=1),delimiter=",",header="instance (id),count",comments='')
           return yvalues


def read_dataset(trainfile, testfile):
	'''
	Reads the input data from train and test files and 
	Returns the matrices Xtrain : [N X D] and Ytrain : [N X 1] and Xtest : [M X D] 
	where D is number of features and N is the number of train rows and M is the number of test rows
	'''
	xtrain = []
	ytrain = []
	xtest = []

	with open(trainfile,'r') as f:
          reader = csv.reader(f,delimiter=',')
          next(reader, None)
          for row in reader:
              #if(row[-1]== '0'):
               #   print(row[-1])
                #  continue
              xtrain.append(row[:-1])
              ytrain.append(row[-1])

	with open(testfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtest.append(row)
     
	return np.array(xtrain), np.array(ytrain), np.array(xtest)

def preprocess_dataset(xdata, ydata=None):
    
    npnoisywind = np.vectorize(correctnoisyspeed)
    xdata[:,-1] = npnoisywind(xdata[:,-1])
    xdata[:,-2] = npnoisywind(xdata[:,-2])
    xdata=np.delete(xdata,0,1)
    xdata = np.append(xdata,splitdate(xdata[:,0]),axis=1)

    xdata = np.delete(xdata,0,1)
    npweeknum = np.vectorize(getweeknum)

    xdata[:,3]= npweeknum(xdata[:,3])
    xdata = xdata.astype(np.float)
    bias = np.ones(shape=(len(xdata),1))

    xdata=np.append(xdata,zerooneencode(xdata[:,3],7),axis=1)
    xdata =np.append(xdata,zerooneencode(xdata[:,0],4),axis=1)
    xdata=np.append(xdata,zerooneencode(xdata[:,1],24),axis=1)
    xdata=np.append(xdata,zerooneencode(xdata[:,5],4),axis=1)

    xdata=np.delete(xdata,5,1)
    xdata= np.delete(xdata,3,1)
    xdata=np.delete(xdata,1,1)
    xdata=np.delete(xdata,0,1)
    xdata = np.c_[bias, xdata]

    if(ydata is not None):
        ydata = ydata.astype(np.float)
        #ydata = np.log10(ydata)
        return xdata,ydata
    
    return xdata

dictionary_of_losses = {
	'mse':(mean_squared_loss, mean_squared_gradient),
	'mae':(mean_absolute_loss, mean_absolute_gradient),
	'rmse':(root_mean_squared_loss, root_mean_squared_gradient),
	'logcosh':(mean_log_cosh_loss, mean_log_cosh_gradient),
}

def main():
    xtrain, ytrain, xtest = read_dataset(args.train_file, args.test_file)

    xtrainprocessed, ytrainprocessed = preprocess_dataset(xtrain, ytrain)
    xtestprocessed = preprocess_dataset(xtest)
    model = LinearRegressor(np.size(xtrainprocessed,1))

    loss_fn, loss_grad = dictionary_of_losses[args.loss]

    model.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, args.epoch, args.lr)

    ytest = model.predict(xtestprocessed)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
     # learning rate = 0.1 , epoch =2500 for kaggle submission
	parser.add_argument('--loss', default='mse', choices=['mse','mae','rmse','logcosh'], help='loss function')
	parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
	parser.add_argument('--epoch', default=2500, type=int, help='number of epochs')
	parser.add_argument('--train_file', type=str, help='location of the training file')
	parser.add_argument('--test_file', type=str, help='location of the test file')

	args = parser.parse_args()

	main()

