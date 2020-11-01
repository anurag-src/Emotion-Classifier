import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.layers import Dense

dataset = pd.read_csv('faces.csv',header=0)

# X & Y separation
n = 25
X = dataset.iloc[:,0:n]
Y = dataset.iloc[:,709]

# Scaling
scale = StandardScaler()
X = scale.fit_transform(X)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 20)

m = x_train.shape[1]
print(m)

# ANN
NN = Sequential()

# first layer
NN.add(Dense(100, activation='relu', kernel_initializer='random_normal', input_dim = m))
# Second Layer
NN.add(Dense(50, activation='relu', kernel_initializer='random_normal'))
# Third Layer
NN.add(Dense(10, activation='relu', kernel_initializer='random_normal'))
#Output Layer
NN.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# Compiling Neural Net
NN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
NN.fit(x_train, y_train, batch_size=10, epochs=100)

# loss value
perf = NN.evaluate(x_train, y_train)
loss = perf[0]
accuracy = perf[1]
print(loss)

# Prediction Accuracy
y_pred = NN.predict(x_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
total_pred = cm[0,0] + cm[1,1] + cm[0,1] + cm[1,0]
pred_accu = ((cm[0,0] + cm[1,1])/total_pred)*100
print(pred_accu)