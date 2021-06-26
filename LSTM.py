from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, Conv1D
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from keras import backend as K
from sklearn import preprocessing
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import keras
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1): #:time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1): #time_step - 1):
        a = dataset[i:(i + look_back), 0] #time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0]) #time_step, 0])
    return np.array(dataX), np.array(dataY)

# Fix Random seed for reproductibility
np.random.seed(7)

# Input Data Train dan Data Test Bahasa Isyarat
dataframe = pd.read_csv('datatrainingisyarat.csv', sep=';')
dataset = dataframe.values
dataset = dataset.astype("float32")
#print(dataset)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.6)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# reshape into X=t and Y=t
look_back = 1 #time_step = 1
trainX, trainY = create_dataset(train, look_back)#time_step)
testX, testY = create_dataset(test, look_back)#time_step)
#print(testX.shape), print(testY.shape)
#print(trainX.shape), print(trainY.shape)

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX  = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX,testX)

# # create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))#time_step)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
#
# # make predictions
trainPredict = model.predict(trainX)
print(trainPredict)
testPredict = model.predict(testX)
print(testPredict)

# create empty table with 12 fields
# trainPredict_dataset_like = np.zeros(shape=(len(trainPredict), 23) )
# put the predicted values in the right field
# trainPredict_dataset_like[:,0] = trainPredict[:,0]
# inverse transform and then select the right field
#trainPredict = scaler.inverse_transform(trainPredict_dataset_like)[:,0]
# trainPredict = np.array(trainPredict).reshape(-1,1,)
# print(trainPredict)

# invert predictions (big problem start here
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# # calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# print(trainX, testX)
# print(testX, testY)


# print(traindata.head)

# testdata = pd.read_csv('datatestisyarat.csv', sep=';')
# print(testdata.shape)
# print(testdata.head)


# Convolutional Neural Network
# model = Sequential()
# model.add(Conv1D())
