# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:14:18 2022

@author: user
"""

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')



#get the data set
df=pd.read_csv('B:/Netty/TSLA.csv')

#number of missing values in each column 
df.isnull().sum()

#printing the first 10 rows
df.head(10)

#number of rows and columns
df.shape

#visual the closing price history
plt.figure(figsize=(16,8))
plt.title('STOCK PRICE PREDICTION')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

#create a new dataframe with only the close colunmn
data=df.filter(['Close'])

#convert the dataframe to a numpy array
dataset=data.values


#Get the numbers of rws to train the model on
training_data_len=math.ceil(len(dataset)* .8)
training_data_len


#scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

scaled_data

#create the training data set
train_data=scaled_data[0:training_data_len ,:]

#split  the data i nto x and y train data
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
 x_train.append(train_data[i-60:i,0])
 y_train.append(train_data[i, 0])
if(i<=60):

  print(x_train)
  print(y_train)
  print() 

# convert the X_train and y_train to numpy arrays
x_train, y_train=np.array(x_train),np.array(y_train)

#Reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1], 1))


x_train.shape

model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],  1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


#Compile the model using linear logistic regression
model.compile(optimizer='adam',loss='mean_squared_error')

#train the model
model.fit(x_train,y_train, batch_size=1,epochs=1)

#create the testing data set
#Create a new array contraining scaled values from index 1543 t2003
test_data=scaled_data[training_data_len-60:,:]
#create the data sets x_test and y_test
x_test=[]
y_test=dataset[training_data_len: ,:]
for i in range(60,len(test_data)):
   x_test.append(test_data[i-60:i, 0])

#convert the data to a numpy array
x_test=np.array(x_test)

#reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


#get the models predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)


#get the root mean squared error
rmse=np.sqrt(np.mean(predictions-y_test)**2)
rmse


#plot the data
train=data[:training_data_len]
valid=data[training_data_len:]
valid["Predictions"]=predictions
plt.figure(figsize=(16,8))
plt.title('STOCK PRICE  PREDICTION')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price VALUE($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'], loc='lower right')
plt.show()



apple_quote=pd.read_csv('B:/Netty/TSLA.csv')
#create the new dataframe
new_df= apple_quote.filter(['Close'])
last_60_days=new_df[- 60:].values
#create an emtty list
last_60_days_scaled=scaler.transform(last_60_days)
x_test=[]
#Append the 60 days
x_test.append(last_60_days_scaled)
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1], 1))
pred_price=model.predict(x_test)
pred_price=scaler.inverse_transform(pred_price)

print(pred_price)

#show the valid and predicted prices
valid
r2_score(y_test, predictions)*100