from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
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

## Input Data Train dan Data Test Bahasa Isyarat

dataframe = pd.read_csv('datatrainingisyaratnumber.csv', sep=';')
dataset = dataframe.values
dataset = dataset.astype("float32")
#print(dataset)

#print(len(train), len(test))

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# print(dataset.shape)

# split into train and test sets
train_size = int(len(dataset) * 0.6)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t
look_back = 1
#time_step = 23
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#print(testX.shape), print(testY.shape)
#print(trainX.shape), print(trainY.shape)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX  = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#print(trainX,testX)

# # create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))#time_step)))
model.add(Dropout(0.2))
model.add(Dense(23, input_dim=1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=500, batch_size=5, validation_split=0.33)

# # make predictions
a = model.predict(trainX)
#print(a.shape)
b = model.predict(testX)
print(b.shape)

# invert predictions (big problem start here
trainPredict = scaler.inverse_transform(a)
#print(a.shape)

trainY_extended = np.zeros((len(trainY),23))
trainY_extended[:,2]=trainY
#print(trainY_extended[:,2].shape)
trainY = scaler.inverse_transform(trainY_extended)[:,2]

testPredict = scaler.inverse_transform(b)
#print(b.shape)

testY_extended = np.zeros((len(testY),23))
testY_extended[:,2]=testY
testY = scaler.inverse_transform(testY_extended)[:,2]
#print(testY_extended[:,2].shape)

# # calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# # shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# # shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# Plot predict train and predict test
predict_train, = plt.plot(trainPredictPlot[:,2])
predict_test,= plt.plot(testPredictPlot[:,2],linestyle='--')
plt.title('Hasil DataProcessing Bahasa Isyarat')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend([predict_train,predict_test], ['training', 'test'], loc='upper right')
plt.show()


# # plot baseline and predictions
serie, = plt.plot(scaler.inverse_transform(dataset)[:,2], linestyle='--')
predict_train,= plt.plot(trainPredictPlot[:,2])
predict_test,= plt.plot(testPredictPlot[:,2])
plt.title('Hasil DataProcessing Bahasa Isyarat')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend([serie,predict_train,predict_test], ['serie','training', 'test'], loc='upper right')
plt.show()

#trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
#testY  = np.reshape(testY, (testY.shape[0], 1, testY.shape[1]))
# create empty table with 12 fields
# trainPredict_dataset_like = np.zeros(shape=(len(trainPredict), 23) )
# put the predicted values in the right field
# trainPredict_dataset_like[:,0] = trainPredict[:,0]
# inverse transform and then select the right field
#trainPredict = scaler.inverse_transform(trainPredict_dataset_like)[:,0]
# trainPredict = np.array(trainPredict).reshape(-1,1,)
# print(trainPredict)

# print(trainX, testX)
# print(testX, testY)


# print(traindata.head)

# testdata = pd.read_csv('datatestisyarat.csv', sep=';')
# print(testdata.shape)
# print(testdata.head)


# Convolutional Neural Network
# model = Sequential()
# model.add(Conv1D())
